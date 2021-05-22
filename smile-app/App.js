import React, { useState } from 'react';
import { View } from 'react-native';
import CameraSmile from './camera.js';
import Testing from './testing';

//Root of the app with the initialisation of all providers

export default function App() {

  const [faces, setFaces] = useState([]);

  return (
    <View>
        <CameraSmile set={setFaces}></CameraSmile>
        <Testing faces={faces}></Testing>
    </View>

  );
}