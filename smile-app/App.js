

import { Button, Input } from 'react-native-elements';
import Svg, {Rect} from 'react-native-svg';
import * as tf from '@tensorflow/tfjs';
import { fetch, bundleResourceIO } from '@tensorflow/tfjs-react-native';
import * as blazeface from '@tensorflow-models/blazeface';
import * as jpeg from 'jpeg-js'
import MyComponent from './compo'
import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, TouchableOpacity } from 'react-native';
import { Camera } from 'expo-camera';
import * as FaceDetector from 'expo-face-detector'
import { cameraWithTensors } from '@tensorflow/tfjs-react-native';


export default function App() {
  const [hasPermission, setHasPermission] = useState(null);
  const [type, setType] = useState(Camera.Constants.Type.back);
  const [faces, setFaces] = useState([]);

  useEffect(() => {

    (async () => {
      await tf.ready();
      console.log('ready')
      const { status } = await Camera.requestPermissionsAsync();
      setHasPermission(true);
      
      
    })();
  }, []);



  if (hasPermission) {
  return(
    <MyComponent></MyComponent>
  ); }
  else {
    return <Text>No access to camera</Text>;
  }}







  /*  const handleFacesDetected = ({faces}) => {
    if(faces.length > 0){
        setFaces({ faces });
        
    }
  };
  return (
    <View style={styles.container}>
      <Camera style={styles.camera} type={type}
        onFacesDetected={handleFacesDetected}
       faceDetectorSettings={{
        mode: FaceDetector.Constants.Mode.fast,
        detectLandmarks: FaceDetector.Constants.Mode.none,
        runClassifications: FaceDetector.Constants.Mode.none,
}}>
        <View style={styles.buttonContainer}>
          <TouchableOpacity
            style={styles.button}
            onPress={() => {
              setType(
                type === Camera.Constants.Type.back
                  ? Camera.Constants.Type.front
                  : Camera.Constants.Type.back
              );
            }}>
            <Text style={styles.text}> Flip </Text>
          </TouchableOpacity>
        </View>
      </Camera>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  buttonContainer: {
    flex: 1,
    backgroundColor: 'transparent',
    flexDirection: 'row',
    margin: 20,
  },
  button: {
    flex: 0.1,
    alignSelf: 'flex-end',
    alignItems: 'center',
  },
  text: {
    fontSize: 18,
    color: 'white',
  },
});

export default class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      isTfReady: false,
    };
  }
 
  async componentDidMount() {
    // Wait for tf to be ready.
    await tf.ready();
    // Signal to the app that tensorflow.js can now be used.
    console.log("Tensorflow initialised")
    this.setState({
      isTfReady: true,
    });
  }
 
 
  render() {
    return(
       <View>
      <Text>test</Text>
    </View>
    )
   
  }
}*/