import React, { useState } from 'react';
import { ActivityIndicator, Text, View, ScrollView, StyleSheet, Button, Platform, LogBox } from 'react-native';
import Svg, {Rect, Image as Zimage, G} from 'react-native-svg';
import Constants from 'expo-constants';
//Root of the app with the initialisation of all providers
import angry from './emotions/angry.png';
import disgust from './emotions/disgust.png';
import fear from './emotions/fear.png';
import happy from './emotions/happy.png';
import neutral from './emotions/neutral.png';  
import sad from './emotions/sad.png';
import surprise from './emotions/surprise.png';
export default function Testing(props) {  
/**
 *                 left
 * 
 * right           jaune
 * 
 * d1 right/jaune = xleft - xright
 * d2 left/jaune  = yright - yleft
 * arctan d2 l/j / d1 r/j = theta
 * 
 * "landmarks": Array [
      Array [
        29.62659053504467, right x
        95.04497796297073,       y
      ],
      Array [
        40.573139011859894, left
        92.18776822090149,
      ],
      Array [
        42.443469405174255,
        102.03396007418633,
      ],
      Array [
        41.375415205955505,
        110.94806864857674,
      ],
      Array [
        13.263086318969727,
        102.39055790007114,
      ],
      Array [
        37.69575674831867,
        95.14466747641563,
      ],
    ]
 */
//Layer for the emoticons
  return (
    <View>
        <Svg style={styles.camera}>
          
          {
            props.faces.map((face)=>{ 
              let xLeft = face.location.landmarks[1][0]
              let yLeft = face.location.landmarks[1][1]
              let xRight = face.location.landmarks[0][0]
              let yRight = face.location.landmarks[0][1]
              let d1 = xLeft - xRight
              let d2 = yRight -yLeft
              let deg = (180/Math.PI)*Math.atan(d2/d1)
              
              return (
                
                <G key={face.id} rotation={deg} origin={(face.location.bottomRight[0] - face.location.topLeft[0])+","+(face.location.bottomRight[1] - face.location.topLeft[1])}  x={face.location.topLeft[0]*2} y={face.location.topLeft[1]*2}>
                
                <Zimage 
                  key={face.id}
                  x="0"
                  y="0"
                  width={(face.location.bottomRight[0] - face.location.topLeft[0])*2}
                  height={(face.location.bottomRight[1] - face.location.topLeft[1])*2}
                  href={face.emotion}
                  
                /></G>
              )
            })
          }   
        </Svg>
    </View>

  );
}

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
      top: 50,
      width: 600 / 2,
      height: 800 / 2,
      zIndex: 1,
  
    },
  
  
  
  });