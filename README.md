# HOLMES - Criminal Detection Platform

![HOLMES Logo](static/img/logo.png)

## About the Application

**HOLMES** is a sophisticated facial recognition application designed to identify criminals by matching their faces against a database of known individuals. This web application provides an efficient and reliable solution for law enforcement agencies to quickly verify identities and access criminal records.

### Key Features

- **Facial Recognition**: Advanced facial recognition technology to identify individuals in real-time
- **Criminal Database**: Maintain and search through a comprehensive database of criminal profiles
- **Modern UI/UX**: Sleek, responsive interface with dark/light mode support
- **Secure Detection**: Fast and accurate matching of facial features
- **History Tracking**: Log and review all detection activities
- **User-friendly Management**: Easy-to-use interfaces for adding, editing, and deleting profiles

## UI/UX Features

The application has been redesigned with a modern and user-friendly interface that includes:

- **Responsive Design**: Fully optimized for desktop, tablet, and mobile devices
- **Dark/Light Mode**: Toggle between themes based on user preference
- **Smooth Animations**: Micro-interactions and transitions for a polished feel
- **Accessibility**: WCAG-compliant design elements for better accessibility
- **Intuitive Navigation**: Streamlined user flows and clear information architecture
- **Modern Components**: Cards, tables, and forms with consistent styling

## Technologies Used

- **Backend**: Python, Django web framework
- **Face Recognition**: OpenCV, face_recognition API
- **Frontend**: HTML, Tailwind CSS, JavaScript
- **UI Framework**: Tailwind CSS with custom components
- **Icons**: Font Awesome
- **Fonts**: Google Fonts (Inter)

## Requirements

- Python version 3.9.13 or higher

## Installation

1. Clone the repository

2. Install required packages:
```
pip install --upgrade pip
pip install -r requirements.txt
```

3. For face recognition functionality, you'll need to install dlib:

   For Python 3.9:
   ```
   pip install https://github.com/Murtaza-Saeed/dlib/raw/master/dlib-19.22.1-cp39-cp39-win_amd64.whl
   ```

   For other Python versions, see the original installation instructions.

4. Run the server:
```
python manage.py runserver
```

5. Open your browser and navigate to:
```
http://127.0.0.1:8000/
```

## Usage Guide

### Home Page
The home page provides quick access to the main functions of the application and displays the detection history.

### Criminal Profiles
View, add, edit, and delete criminal profiles from the database.

### Detection
Use your camera to scan and identify individuals in real-time.

## Screenshots And Demo
Demo:https://photos.app.goo.gl/GG41JhxycK9x7kWf9

### Dark Mode
![Dark Mode](Screenshots/homepage.jpg)

### add new criminal
![add new criminal](Screenshots/add_new_criminal.jpg)

### Criminal Profiles
![Profiles](Screenshots/criminal_profiles.jpg)

### Criminal Details
![Details](Screenshots/criminal_identified.jpg)

## Contributing

Contributions to improve HOLMES are welcome. Please feel free to fork the repository and submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Special thanks to all contributors who have helped develop and improve this application
- Face recognition libraries and their maintainers
