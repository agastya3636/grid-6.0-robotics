import React from 'react';
import { Link } from 'react-router-dom'; // Import Link from react-router-dom
import './style/NavBar.css'; // You can style the navbar with CSS here

const NavBar = () => {
  return (
    <nav className="navbar">
      <h1 className="navbar-heading">Flipkart Grid 2024</h1>
      <ul className="nav-list">
        <li className="nav-item"><Link to="/description">Description of product</Link></li>
        <li className="nav-item"><Link to="/specific-info">Specific info</Link></li>
        <li className="nav-item"><Link to="/object-detection">Object Detection</Link></li>
        <li className="nav-item"><Link to="/freshness">Freshness</Link></li>
      </ul>
    </nav>
  );
};

export default NavBar;
