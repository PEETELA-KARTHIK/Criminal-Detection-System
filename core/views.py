from django.shortcuts import render, HttpResponse, redirect
from .models import *
from .forms import *
import face_recognition
import cv2
import numpy as np
from django.db.models import Q
import os
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable for last detected face
last_face = None

last_face = 'no_face'
current_path = os.path.dirname(__file__)
face_list_file = os.path.join(current_path, 'face_list.txt')

# for implementing history
def home(request):
    scanned = LastFace.objects.all().order_by('date').reverse()
    context = {
        'scanned': scanned,
    }
    return render(request, 'core/home.html', context)


def ajax(request):
    last_face = LastFace.objects.last()
    context = {
        'last_face': last_face
    }
    return render(request, 'core/ajax.html', context)

# for implementing face recognition
def scan(request):
    global last_face
    try:
        # Initialize face recognition variables
        found_face_encodings = []
        found_face_names = []

        # Get all profiles and their face encodings
        profiles = Profile.objects.all()
        for profile in profiles:
            try:
                person = profile.image
                image_of_person = face_recognition.load_image_file(f'media/{person}')
                person_face_encoding = face_recognition.face_encodings(image_of_person)[0]
                found_face_encodings.append(person_face_encoding)
                found_face_names.append(f'{person}'[:-4])
            except Exception as e:
                logger.error(f"Error processing profile {profile.id}: {str(e)}")
                continue

        if not found_face_encodings:
            logger.warning("No face encodings found in profiles")
            return HttpResponse('No profiles found with valid face images')

        # Initialize video capture
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            logger.error("Could not open video capture")
            return HttpResponse('Could not open camera')

        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True

        while True:
            ret, frame = video_capture.read()
            if not ret:
                logger.error("Failed to grab frame")
                break

            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            if process_this_frame:
                # Find faces in current frame
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # Set tolerance for face comparison (lower = more strict)
                    matches = face_recognition.compare_faces(found_face_encodings, face_encoding, tolerance=0.6)
                    name = "Criminal not found in records"

                    if True in matches:
                        face_distances = face_recognition.face_distance(found_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        
                        if matches[best_match_index]:
                            name = found_face_names[best_match_index]
                            
                            try:
                                profile = Profile.objects.get(Q(image__icontains=name))
                                if not profile.present:
                                    profile.present = True
                                    profile.save()
                                
                                if last_face != name:
                                    last_face_obj = LastFace(last_face=name)
                                    last_face_obj.save()
                                    last_face = name
                            except Profile.DoesNotExist:
                                logger.error(f"Profile not found for {name}")
                            except Exception as e:
                                logger.error(f"Error updating profile: {str(e)}")

                    face_names.append(name)

            process_this_frame = not process_this_frame

            # Draw results on frame
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw box around face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw label below face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

            # Display the frame
            cv2.imshow('Face detection - Press q to shut camera', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Clean up
        video_capture.release()
        cv2.destroyAllWindows()
        return HttpResponse('Scanner closed')

    except Exception as e:
        logger.error(f"Error in scan function: {str(e)}")
        if 'video_capture' in locals():
            video_capture.release()
        cv2.destroyAllWindows()
        return HttpResponse(f'Error occurred: {str(e)}')


def profiles(request):
    profiles = Profile.objects.all()
    context = {
        'profiles': profiles
    }
    return render(request, 'core/profiles.html', context)


def details(request):
    try:
        last_face = LastFace.objects.last()
        profile = Profile.objects.get(Q(image__icontains=last_face))
    except:
        last_face = None
        profile = None

    context = {
        'profile': profile,
        'last_face': last_face
    }
    return render(request, 'core/details.html', context)

# to take input in the form 
def add_profile(request):
    form = ProfileForm
    if request.method == 'POST':
        form = ProfileForm(request.POST,request.FILES)
        if form.is_valid():
            form.save()
            return redirect('profiles')
    context={'form':form}
    return render(request,'core/add_profile.html',context)

# to take input in the edit form
def edit_profile(request,id):
    profile = Profile.objects.get(id=id)
    form = ProfileForm(instance=profile)
    if request.method == 'POST':
        form = ProfileForm(request.POST,request.FILES,instance=profile)
        if form.is_valid():
            form.save()
            return redirect('profiles')
    context={'form':form}
    return render(request,'core/add_profile.html',context)

# to delete criminal profile
def delete_profile(request,id):
    profile = Profile.objects.get(id=id)
    profile.delete()
    return redirect('profiles')

# to delete history
def clear_history(request):
    history = LastFace.objects.all()
    history.delete()
    return redirect('home')



