# Local Setup Guide for HOLMES

This guide will help you set up and run the HOLMES project locally on your machine.

## Prerequisites

1. Python 3.9 or higher
2. Git (optional, for cloning the repository)
3. Webcam (for facial recognition features)

## Installation Steps

1. **Create a Virtual Environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install dlib**
   - For Windows:
     - Download the appropriate wheel file from [here](https://github.com/jloh02/dlib/releases)
     - Install using: `pip install path/to/dlib_wheel_file.whl`
   - For Linux:
     ```bash
     sudo apt-get install cmake
     sudo apt-get install libdlib-dev
     pip install dlib
     ```
   - For Mac:
     ```bash
     brew install cmake
     pip install dlib
     ```

4. **Set Up the Database**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. **Create a Superuser**
   ```bash
   python manage.py createsuperuser
   ```

6. **Install Tailwind CSS**
   ```bash
   python manage.py tailwind install
   python manage.py tailwind build
   ```

## Running the Application

1. **Start the Development Server**
   ```bash
   python manage.py runserver
   ```

2. **Access the Application**
   - Open your web browser and go to: `http://127.0.0.1:8000`
   - The admin interface will be available at: `http://127.0.0.1:8000/admin`

## Important Notes

1. Make sure your webcam is connected and accessible
2. The application requires camera permissions to work properly
3. For best results, ensure good lighting conditions when using facial recognition
4. The database is SQLite by default, which is suitable for local development

## Troubleshooting

1. **Camera Issues**
   - Check if your webcam is properly connected
   - Ensure you've granted camera permissions to your browser
   - Try using a different browser if issues persist

2. **Database Issues**
   - If you get database errors, try:
     ```bash
     python manage.py makemigrations
     python manage.py migrate
     ```

3. **Tailwind CSS Issues**
   - If styles aren't loading, run:
     ```bash
     python manage.py tailwind build
     ```

## Development Tips

1. Keep the development server running while making changes
2. Use the Django admin interface to manage criminal profiles
3. The application will automatically reload when you make changes to Python files
4. For Tailwind CSS changes, you need to rebuild using `python manage.py tailwind build`

## Security Note

This is a development setup. For production use, you should:
- Use a proper database (like PostgreSQL)
- Set up proper security measures
- Use environment variables for sensitive data
- Configure proper static file serving 