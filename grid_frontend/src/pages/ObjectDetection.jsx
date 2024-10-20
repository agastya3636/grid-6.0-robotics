import React from 'react';
import '../components/style/ObjectDetection.css'; // Adjusted path to the CSS file
import ImageUpload from '../components/ImageUpload.jsx'; // Import the ImageUpload component

const ObjectDetection = () => {
  return (
    <div className="wrap">
      <div className="image">
        {/* Add the ImageUpload component */}
        <ImageUpload endpoint='caption' />
      </div>
      <div className="description-container">
        <h2>Object Detection</h2>
        <p>This page showcases object detection features and functionalities.</p>
      </div>
    </div>
  );
};

export default ObjectDetection;
