# Dynamic-Multi-University-Chatbot-Framework

## Steps to Use:

System Requirements
Ensure that your system meets the following requirements:
•	Operating System: Windows, macOS, or Linux
•	Python: Version 3.8 or higher
•	Node.js: Version 16 or higher
•	npm: Version 7 or higher
•	Git: Installed for cloning the repository
________________________________________
Installation and Setup
1. Clone the Repository
Open a terminal and clone the repository:
bash
Copy code
[https://github.com/aditya11997/Yeshiva_University_KatzBot.git](https://github.com/aditya11997/Dynamic-Multi-University-Chatbot-Framework.git)
cd my-app
________________________________________
3. Set Up Python Virtual Environment
1.	Create a virtual environment (in the my-app directory):
bash
Copy code
python3 -m venv venv
2.	Activate the virtual environment:
o	On macOS/Linux:
bash
Copy code
source venv/bin/activate
o	On Windows:
bash
Copy code
venv\Scripts\activate
________________________________________
3. Install Python Dependencies
1.	Ensure the virtual environment is activated.
2.	Install required Python packages:
bash
Copy code
pip install -r requirements.txt
This will install Flask, BeautifulSoup, and other necessary dependencies.
________________________________________
4. Set Up Frontend Dependencies
1.	Navigate to the my-app folder:
bash
Copy code
cd my-app
2.	Install required npm packages:
bash
Copy code
npm install
________________________________________
Running the Application
1. Start Flask Backend
1.	Open a terminal, navigate to the my-app folder, and activate the virtual environment:
o	On macOS/Linux:
bash
Copy code
cd my-app
source venv/bin/activate
o	On Windows:
bash
Copy code
cd my-app
venv\Scripts\activate
2.	Start the Flask backend:
bash
Copy code
flask run
3.	Verify the backend is running: You should see output indicating the Flask server is running on http://127.0.0.1:5000.
________________________________________
2. Start React Frontend
1.	Open another terminal and navigate to the my-app folder:
bash
Copy code
cd my-app
2.	Start the React frontend:
bash
Copy code
npm start
3.	Verify the frontend is running: Your default browser should open and display the chatbot UI. If it doesn’t, you can access it manually at:
arduino
Copy code
http://localhost:3000
________________________________________
Accessing the Chatbot Locally
To interact with the chatbot, open the following URLs in your browser:
•	Main Chatbot Interface:
arduino
Copy code
http://localhost:3000/<university-name>
Replace <university-name> with the name of the registered university, e.g., http://localhost:3000/yeshiva-university.
•	Admin Panel:
bash
Copy code
http://localhost:3000/admin
Log in with valid credentials to manage chatbot configurations.
•	Login Page:
bash
Copy code
http://localhost:3000/login
Use this URL to authenticate yourself as an admin.
________________________________________
FAQ
1. What if Flask doesn't run?
•	Ensure the virtual environment is activated.
•	Check if flask is installed:
bash
Copy code
pip show flask
•	If Flask is not installed, run:
bash
Copy code
pip install flask
2. What if npm start fails?
•	Ensure Node.js and npm are installed. Run:
bash
Copy code
node -v
npm -v
•	If not installed, download and install Node.js from https://nodejs.org/.
3. What URLs should I hit for unregistered universities?
•	If no university is registered, navigate to:
arduino
Copy code
http://localhost:3000/<university-name>
You will see an interface to upload a sitemap or textual data for registration.

