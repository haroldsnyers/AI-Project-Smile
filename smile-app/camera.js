import React, { useState, useEffect, Component } from 'react';
import { ActivityIndicator, Text, View, ScrollView, StyleSheet, Button, Platform, LogBox } from 'react-native';
import Constants from 'expo-constants';
import RNPickerSelect from '@react-native-picker/picker';
import { Chevron } from 'react-native-shapes';
import Svg, {Rect, Image as Zimage} from 'react-native-svg';
//Permissions
import * as Permissions from 'expo-permissions';


//camera
import { Camera } from 'expo-camera';


//tensorflow
import * as tf from '@tensorflow/tfjs';
import * as blazeface from '@tensorflow-models/blazeface';
import { cameraWithTensors, asyncStorageIO, bundleResourceIO,detectGLCapabilities} from '@tensorflow/tfjs-react-native';

import angry from './emotions/angry.png';
import disgust from './emotions/disgust.png';
import fear from './emotions/fear.png';
import happy from './emotions/happy.png';
import neutral from './emotions/neutral.png';  
import sad from './emotions/sad.png';
import surprise from './emotions/surprise.png';

//disable yellow warnings on EXPO client!
LogBox.ignoreAllLogs(true)
const TensorCamera = cameraWithTensors(Camera);
const textureDims = Platform.OS === "ios" ? { width: 1080, height: 1920 } : { width: 1600, height: 1200 };
const tensorDims = { width: 152 , height: 200 };
let requestAnimationFrameId = 0;
export default class CameraSmile extends Component {
  constructor(props) {
    super(props);
    this.state = {
        predictionFound : false,
        hasPermission : null,
        maskDetector : "",
        blazefaceModel: null,
        frameworkReady: false
    }
  }
  //initilization
  componentDidMount(){
    if (!this.state.frameworkReady) {
      (async () => {

        //check permissions
        const { status } = await Camera.requestPermissionsAsync();
        console.log(`permissions status: ${status}`);

        //we must always wait for the Tensorflow API to be ready before any TF operation...
        await tf.ready();       
        let model =await this.loadBlazefaceModel();
        const modelJson = await require("./assets/models/model.json");
        const modelWeight = await require("./assets/models/group1-shard.bin");
        const maskDetector = await tf.loadLayersModel(bundleResourceIO(modelJson,modelWeight));
        this.setState({maskDetector:maskDetector,frameworkReady:true, hasPermission:status === 'granted',blazefaceModel:model})
      })();
    }
    
  }
  componentWillUnmount(){
    cancelAnimationFrame(requestAnimationFrameId);
  }

  shouldComponentUpdate(){
    return false
  }

  loadBlazefaceModel = async () => {
    const model = await blazeface.load();
    return model;
  }

  bestEmotion(data){
    let emotion = [angry,disgust,fear,happy,sad,surprise,neutral]
    let i = 0
    let max = 0
    let id = 0
      data.forEach(element => {
        if(element> max){
          id = i
          max = element
        }
        i++
      });
    return emotion[id]
  }

  getPredictionBlaze = async (imageTensor) => {
    if (!imageTensor) { return; }
    if (this.state.blazefaceModel != null & true) {
      this.props.set([])
    const faces = await this.state.blazefaceModel.estimateFaces(imageTensor, false)
    
    //const prediction = await blazefaceModel.predict(tensor).data()
    if (!faces || faces.length === 0) { return; }

    if(faces.length>0){
          cancelAnimationFrame(requestAnimationFrameId);
        }

    var tempArray=[]
    //console.log(faces)
    for (let i=0;i<faces.length;i++){
      try {
        let width = parseInt((faces[i].bottomRight[1] - faces[i].topLeft[1]))
        let height = parseInt((faces[i].bottomRight[0] - faces[i].topLeft[0]))
        //console.log("[+] facetensor init")
        let faceTensor=imageTensor.slice([parseInt(faces[i].topLeft[1]),parseInt(faces[i].topLeft[0]),0],[width,height,1])
        //console.log("[+] facetensor resize")
        faceTensor = faceTensor.resizeBilinear([48,48]).reshape([1,48,48,1])
        //console.log("[+] facetensor result")
        let result = await this.state.maskDetector.predict(faceTensor).data()
        let emotion = this.bestEmotion(result)
        tempArray.push({
          id:i,
          location:faces[i],
          emotion: emotion
        })
      } catch (error) {
        this.props.set([])
      }
      
  }
 this.props.set(tempArray)
    
  }
  }
  handleCameraStream = (imageAsTensors) => {
    let i = 0;
    const loop = async () => {
      const nextImageTensor = await imageAsTensors.next().value;
      //await getPredictionMobile(nextImageTensor);
      if(i == 1){
        await this.getPredictionBlaze(nextImageTensor);
        i=0
      }
      i += 1
      requestAnimationFrameId = requestAnimationFrame(loop);
    };
    if (!this.state.predictionFound) loop();
  }


  /*-----------------------------------------------------------------------
  Helper function to show the Camera View. //={(imageAsTensors) => handleCameraStream(imageAsTensors)}
  
  NOTE: Please note we are using TensorCamera component which is constructed on line: 37 of this function component. This is just a decorated expo.Camera component with extra functionality to stream Tensors, define texture dimensions and other goods. For further research:
  https://js.tensorflow.org/api_react_native/0.2.1/#cameraWithTensors
  -----------------------------------------------------------------------*/
  renderCameraView(){
    return <View>
      <TensorCamera
        style={styles.
          camera}
        type={Camera.Constants.Type.back}
        zoom={0}
        cameraTextureHeight={textureDims.height}
        cameraTextureWidth={textureDims.width}
        resizeHeight={tensorDims.height}
        resizeWidth={tensorDims.width}
        resizeDepth={3}
        onReady={(imageAsTensors) => this.handleCameraStream(imageAsTensors)}
        autorender={true}
      />
      
    </View>;
  }

  render(){
  return (
    <View style={styles.body}>
      
    <View style={{marginBottom:20}}>
        { this.renderCameraView()}
        
      </View>
        
    </View>
  );
}}


const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'flex-start',
    paddingTop: Constants.statusBarHeight,
    backgroundColor: '#E8E8E8',
  },
  body: {
    padding: 5,
    paddingTop: 25
  },
  cameraView: {
    display: 'flex',
    flex: 1,
    flexDirection: 'column',
    justifyContent: 'flex-start',
    alignItems: 'flex-end',
    width: '100%',
    height: '100%',
    paddingTop: 10
  },
  camera: {
    position: 'absolute',
    left: 50,
    top: 100,
    width: 600 / 2,
    height: 800 / 2,
    zIndex: 1,

  },



});



