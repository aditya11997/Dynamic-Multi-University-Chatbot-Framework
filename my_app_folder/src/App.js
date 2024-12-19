import React, { useState, useEffect } from 'react';
import './styles/styles.css';
import './App.css';
import { BrowserRouter as Router, Route, Routes, Navigate, useParams } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import AdminPanel from './components/AdminPanel';
import Login from './components/Login';
import DefaultResponse from './components/DefaultResponse'; // Component to show when university chatbot is not registered.

const App = () => {
  const [darkTheme, setDarkTheme] = useState(false);
  const [registeredUniversities, setRegisteredUniversities] = useState(['pace-university', 'yeshiva-university']);
  const [selectedLanguage, setSelectedLanguage] = useState('en-US');

  const toggleTheme = () => setDarkTheme(prevTheme => !prevTheme);

  // Helper function to check if the user is authenticated
  const isAuthenticated = () => {
    return localStorage.getItem('loggedIn') === 'true';
  };

  // Private Route component
  const PrivateRoute = ({ children }) => {
    return isAuthenticated() ? children : <Navigate to="/login" />;
  };

  // Fetch registered universities dynamically on app load
  useEffect(() => {
    const fetchRegisteredUniversities = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5000/registered-universities');
        const data = await response.json();
        if (response.ok && data.universities) {
          setRegisteredUniversities(data.universities);
        }
      } catch (error) {
        console.error('Error fetching registered universities:', error);
      }
    };

    fetchRegisteredUniversities();
  }, []);

  // Dynamic university chatbot loader
  const UniversityChatbot = () => {
    const { universityName } = useParams();
    console.log(universityName);
    console.log(selectedLanguage)
    const isRegistered = registeredUniversities.includes(universityName.toLowerCase());

    if (!isRegistered) {
      return <DefaultResponse universityName={universityName} />;
    }

    return (
      <>
        <Sidebar 
          universityName={universityName}
          selectedLanguage={selectedLanguage} 
          setSelectedLanguage={setSelectedLanguage}
        />
        <div className="app-content">
          <header className="App-header">
            <h1 style={{ fontSize: '45px' }}>
              {universityName.replace(/-/g, ' ').toUpperCase()} Bot
            </h1>
          </header>
          <ChatInterface universityName={universityName} selectedLanguage={selectedLanguage}/>
        </div>
      </>
    );
  };
  

  return (
    <Router>
      <div className={`App ${darkTheme ? 'dark-theme' : 'light-theme'}`}>
        <div className="theme-toggle">
          <button onClick={toggleTheme}>
            {darkTheme ? 'Light Mode' : 'Dark Mode'}
          </button>
        </div>
        <Routes>
          <Route path="/" element={<Navigate to="/yeshiva-university" />} />
          <Route path="/:universityName" element={<UniversityChatbot />} />
          <Route
            path="/admin"
            element={
              <PrivateRoute>
                <div className="app-content">
                  <header className="App-header">
                    <h1 style={{ fontSize: '45px' }}>Admin Panel</h1>
                  </header>
                  <AdminPanel mode="register" />
                </div>
              </PrivateRoute>
            }
          />
          <Route
            path="/update-university-data"
            element={
              <PrivateRoute>
                <div className="app-content">
                  <header className="App-header">
                    <h1 style={{ fontSize: '45px' }}>Update University Data</h1>
                  </header>
                  <AdminPanel mode="update" />
                </div>
              </PrivateRoute>
            }
          />
          <Route path="/login" element={<Login />} />
          <Route path="*" element={<DefaultResponse universityName="Unknown" />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
