import React, { useState, useEffect } from 'react';
import axios from 'axios';

const AdminPanel = () => {
  const [universities, setUniversities] = useState([]);
  const [selectedUniversity, setSelectedUniversity] = useState('');
  const [manualData, setManualData] = useState('');
  const [responseMessage, setResponseMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Fetch registered universities
  useEffect(() => {
    const fetchUniversities = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:5000/registered-universities');
        setUniversities(response.data.universities || []);
      } catch (error) {
        console.error('Error fetching universities:', error);
        setResponseMessage('Failed to load universities. Please refresh the page.');
      }
    };
    fetchUniversities();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setResponseMessage('');
    if (!selectedUniversity || !manualData.trim()) {
      setResponseMessage('Please select a university and enter valid data.');
      return;
    }
    setIsLoading(true);
    try {
      const response = await axios.post('http://127.0.0.1:5000/update-university-data', {
        university_name: selectedUniversity,
        manual_data: manualData,
      });
      setResponseMessage(response.data.message || 'Knowledge base updated successfully!');
      setManualData('');
    } catch (error) {
      console.error('Error updating data:', error);
      const errorMsg = error.response?.data?.error || 'Failed to update data. Please try again.';
      setResponseMessage(errorMsg);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="admin-panel-container">
      <h2 className="admin-panel-header">Update University Knowledge Base</h2>
      {universities.length === 0 ? (
        <p>No universities registered yet. Please add one first.</p>
      ) : (
        <div className="admin-panel">
          <form onSubmit={handleSubmit}>
            <label>
              Select University:
              <select 
                value={selectedUniversity}
                onChange={(e) => setSelectedUniversity(e.target.value)}
                disabled={isLoading}
              >
                <option value="">-- Select --</option>
                {universities.map((uni, index) => (
                  <option key={index} value={uni}>
                    {uni.replace(/_/g, ' ').toUpperCase()}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Enter Additional Data:
              <textarea
                rows="6"
                value={manualData}
                onChange={(e) => setManualData(e.target.value)}
                placeholder="Enter one document per line"
                disabled={isLoading}
              />
            </label>
            <button type="submit">Update Knowledge Base
              {isLoading ? 'Updating...' : 'Update Knowledge Base'}
            </button>
          </form>
        </div>
      )}
      {responseMessage && <p>{responseMessage}</p>}
    </div>
  );
};

export default AdminPanel;
