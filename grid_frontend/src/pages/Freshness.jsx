import React from 'react';
import '../components/style/Freshness.css'; // Adjusted path to the CSS file
import ImageUpload from '../components/ImageUpload.jsx'; // Import the ImageUpload component

const Freshness = () => {
  return (
    <div className="wrap">
      <div className="image">
        {/* Add the ImageUpload component */}
        <ImageUpload endpoint='predict' />
      </div>
      <div className="description-container">
        <h2>Freshness Information</h2>
        <p>Learn more about the freshness features of our products.</p>
      </div>
    </div>
  );
};

export default Freshness;
