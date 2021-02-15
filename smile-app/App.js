import React, { useState, useEffect }  from 'react';
import { StyleSheet, View,Image ,Text} from 'react-native';
import { Button, Input } from 'react-native-elements';
import Svg, {Rect} from 'react-native-svg';
import * as tf from '@tensorflow/tfjs';
import { fetch, bundleResourceIO } from '@tensorflow/tfjs-react-native';
import * as blazeface from '@tensorflow-models/blazeface';
import * as jpeg from 'jpeg-js'

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
}