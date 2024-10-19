import React, { useState } from 'react';
import './style/ImageUpload.css'; // Adjust path if necessary

const ImageUpload = () => {
  const [image, setImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [message, setMessage] = useState('');
  const [isError, setIsError] = useState(false);

  // Handle the file selection
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreviewUrl(URL.createObjectURL(file)); // Create a preview URL for the image
      setMessage(''); // Clear any previous messages
      setIsError(false); // Reset error state
    }
  };

  // Remove the selected image
  const handleRemoveImage = () => {
    setImage(null);
    setPreviewUrl(''); // Clear the preview URL
    setMessage(''); // Clear the message when removing
  };

  // Submit the image to the backend server
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!image) {
      setMessage('Please select an image to upload.');
      setIsError(true);
      return;
    }

    const formData = new FormData();
    formData.append('image', image);

    try {
      const response = await fetch('http://localhost:5000/upload', { // Change to your backend API URL
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        setMessage('Image uploaded successfully!');
        setIsError(false);
      } else {
        setMessage('Failed to upload image.');
        setIsError(true);
      }
    } catch (error) {
      setMessage('Error uploading image.');
      setIsError(true);
    }
  };

  return (
    <div className="image-upload-container">
      <div className={`upload-area ${!image ? 'empty' : ''}`}>
        {!image ? (
          <div className="choose-file-message">No Image file is chosen.</div>
        ) : (
          <>
            <button className="remove-button" onClick={handleRemoveImage}>Remove Image</button>
            <div className="image-preview">
              <img src={previewUrl} alt="Preview" />
            </div>
            <button className="upload-button" onClick={handleSubmit}>Upload Image</button>
          </>
        )}
        
        {/* File input with label */}
        <div className="file-input-container">
          <input 
            type="file" 
            accept="image/*" 
            id="file-upload" 
            onChange={handleFileChange} 
            style={{ display: 'none' }} // Hide the input element
          />
          <label htmlFor="file-upload" className="file-input-label">
            {image ? "Change the file" : "Choose a file to upload"}
          </label>
        </div>
      </div>

      {/* Display message with appropriate styling */}
      {message && (
        <p className={isError ? 'error-message' : 'success-message'}>{message}</p>
      )}
    </div>
  );
};

export default ImageUpload;
