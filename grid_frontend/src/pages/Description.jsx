import React from 'react';
import '../components/style/Description.css'; // Adjusted path to the CSS file
import ImageUpload from '../components/ImageUpload.jsx'; // Import the ImageUpload component

const Description = () => {
  return (
    <>
   
      <div className='wrap'>
      <div className="image">
        {/* Add the ImageUpload component */}
        <ImageUpload endpoint='ocr'/>
      </div>
      <div className="description-container">
        <h2>Description of Product</h2>
        <p>Here you can find the detailed description of the product.</p>
      </div>
    
      </div>
    </>
  );
};

export default Description;
