import React from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import NavBar from './components/NavBar'; // Updated path to NavBar
import Description from './pages/Description'; // Updated path
import SpecificInfo from './pages/SpecificInfo'; // Updated path
import ObjectDetection from './pages/ObjectDetection'; // Updated path
import Freshness from './pages/Freshness'; // Updated path

const App = () => {
  const location = useLocation(); // Get the current location

  return (
    <div style={styles.app}>
      <NavBar />

      {/* Conditional rendering for welcome message */}
      {location.pathname === '/' && (
        <div style={styles.content}>
          <h2 style={styles.welcome}>Welcome to Flipkart Grid!</h2>
          <p style={styles.subtext}>Explore the best products and offers</p>
        </div>
      )}

      {/* Main content - Always displayed, regardless of the route */}
      <div style={styles.content}>
        <Routes>
          <Route path="/description" element={<Description />} />
          <Route path="/specific-info" element={<SpecificInfo />} />
          <Route path="/object-detection" element={<ObjectDetection />} />
          <Route path="/freshness" element={<Freshness />} />
        </Routes>
      </div>
    </div>
  );
};

const styles = {
  app: {
    width: '100vw',
    height: '100vh',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    textAlign: 'center',
    backgroundColor: '#f4e1d2',
  },
  content: {
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
    flexGrow: 1,
    backgroundColor:'#f4e1d2',
  },
  welcome: {
    fontSize: '48px',
    color: '#b2b2b2', // Flipkart blue
    fontWeight: 'bold',
    marginBottom: '10px',
  },
  subtext: {
    fontSize: '20px',
    color: '#b2b2b2', // A softer text color
  },
};

const AppWrapper = () => (
  <Router>
    <App />
  </Router>
);

export default AppWrapper;
