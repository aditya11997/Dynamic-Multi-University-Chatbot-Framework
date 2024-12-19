import React, { useState } from 'react';
import './styles/styles.css';
import './App.css';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import AdminPanel from './components/AdminPanel'; // Import the Admin Panel Component
import Login from './components/Login'; // Import Login Component
import DefaultResponse from './components/DefaultResponse'; // Component to show when university chatbot is not registered.

const App = () => {
  const [darkTheme, setDarkTheme] = useState(false);

  const toggleTheme = () => setDarkTheme(prevTheme => !prevTheme);

  // Helper function to check if the user is authenticated
  const isAuthenticated = () => {
    return localStorage.getItem('loggedIn') === 'true';
  };

  // Private Route component
  const PrivateRoute = ({ children }) => {
    return isAuthenticated() ? children : <Navigate to="/login" />;
  };

  return (
    <Router>  {/* Wrap the app in Router */}
      <div className={`App ${darkTheme ? 'dark-theme' : 'light-theme'}`}>
        <div className="theme-toggle">
          <button onClick={toggleTheme}>
            {darkTheme ? 'Light Mode' : 'Dark Mode'}
          </button>
        </div>
        <Routes>
          <Route
            path="/"
            element={
              <>
                <Sidebar />
                <div className="app-content">
                  <header className="App-header">
                    <h1 style={{ fontSize: '45px' }}>PaceBot</h1>
                  </header>
                  <ChatInterface />
                </div>
              </>
            }
          />
          <Route
            path="/admin"
            element={
              <PrivateRoute>
                <div className="app-content">
                  <header className="App-header">
                    <h1 style={{ fontSize: '45px' }}>PaceBot Admin Panel</h1>
                  </header>
                  <AdminPanel />
                </div>
              </PrivateRoute>
            }
          />
          <Route path="/login" element={<Login />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
