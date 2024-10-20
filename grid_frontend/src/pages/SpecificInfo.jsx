import React from 'react';
import '../components/style/SpecificInfo.css'; // Adjusted path to the CSS file
import ImageUpload from '../components/ImageUpload.jsx'; // Import the ImageUpload component

const SpecificInfo = () => {
  return (
    <div className="wrap">
      <div className="image">
        {/* Add the ImageUpload component */}
        <ImageUpload endpoint='ocr'/>
      </div>
      <div className="description-container">
        <h2>Specific Information</h2>
        <p>Here you can find specific details about the product.</p>
      </div>
    </div>
  );
};

export default SpecificInfo;
