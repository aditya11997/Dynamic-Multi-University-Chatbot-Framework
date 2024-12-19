import React, { useState } from 'react';
import '../styles/DefaultResponse.css'; // Ensure you have styles for this component

const DefaultResponse = ({ universityName }) => {
  const [sitemapUrl, setSitemapUrl] = useState('');
  const [manualData, setManualData] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setMessage('');

    try {
      const response = await fetch('http://127.0.0.1:5000/register-university', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          university_name: universityName || null,
          sitemap_url: sitemapUrl,
          manual_data: manualData || null,
        }),
      });

      const data = await response.json();
      setLoading(false);
      if (response.ok) {
        setMessage(data.message);
        setSitemapUrl(''); // Clear input fields on success
        setManualData('');
      } else {
        setMessage(data.error || 'An error occurred. Please try again.');
      }
    } catch (error) {
      setLoading(false);
      setMessage('Failed to register university. Please try again.');
    }
  };

  return (
    <div className="default-response-container">
      <div className="default-response">
        <h2>{universityName.replace(/-/g, ' ').toUpperCase()} Chatbot</h2>
        <p>
          The chatbot for this university is not yet registered. Please provide the sitemap URL
          to initiate the registration process.
        </p>
        <form onSubmit={handleSubmit} className="registration-form">
          <div className="form-group">
            <label>Enter Sitemap URL:</label>
            <input
              type="text"
              placeholder="Enter sitemap URL"
              value={sitemapUrl}
              onChange={(e) => setSitemapUrl(e.target.value)}
            />
          </div>
          <div className="form-group">
            <label>Or Enter Textual Data:</label>
            <textarea
              placeholder="Enter manual data here (one line per entry)"
              value={manualData}
              onChange={(e) => setManualData(e.target.value)}
              rows="5"
            ></textarea>
          </div>
          <button type="submit" disabled={loading}>
            {loading ? 'Registering...' : 'Register'}
          </button>
        </form>
        {loading && <div className="spinner">⏳ Processing...</div>}
        {message && <p className="message">{message}</p>}
      </div>
    </div>
  );
};

export default DefaultResponse;
