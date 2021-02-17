import { Camera } from 'expo-camera';
import { cameraWithTensors } from '@tensorflow/tfjs-react-native';
import React, { Component } from 'react';

import * as blazeface from '@tensorflow-models/blazeface';
import * as tf from '@tensorflow/tfjs';
import { ActivityIndicator, Button, StyleSheet, View, Text, Platform } from 'react-native';
import Svg, { Circle, Rect, G, Line } from 'react-native-svg';

const TensorCamera = cameraWithTensors(Camera);

export default class MyComponent extends Component {

    constructor(props) {
        super(props);
        this.state = {
            isLoading: true,
            cameraType: Camera.Constants.Type.front,
            modelName: 'blazeface',
            faceDetector: null,
        };
    }

    async loadBlazefaceModel() {
        const model = await blazeface.load();
        return model;
    }

    async componentDidMount() {
        const [blazefaceModel] =
            await Promise.all([this.loadBlazefaceModel()])
            console.log(blazefaceModel)
            this.setState({
                    isLoading: false,
                    faceDetector: blazefaceModel,
                })


    }

    renderFaces() {
        const { faces } = this.state;
        console.log("je vais tout casser")
        /*if (faces != null) {
            console.log("je vais tout réparer")
            const faceBoxes = faces.map((f, fIndex) => {
                const topLeft = f.topLeft;
                const bottomRight = f.bottomRight;

                const landmarks = (f.landmarks).map((l, lIndex) => {
                    return <Circle
                        key={`landmark_${fIndex}_${lIndex}`}
                        cx={l[0]}
                        cy={l[1]}
                        r='2'
                        strokeWidth='0'
                        fill='blue'
                    />;
                });

                return <G key={`facebox_${fIndex}`}>
                    <Rect
                        x={topLeft[0]}
                        y={topLeft[1]}
                        fill={'red'}
                        fillOpacity={0.2}
                        width={(bottomRight[0] - topLeft[0])}
                        height={(bottomRight[1] - topLeft[1])}
                    />
                    {landmarks}
                </G>;
            });

            const flipHorizontal = Platform.OS === 'ios' ? 1 : -1;
            return <Svg height='100%' width='100%'
                viewBox={`0 0 ${inputTensorWidth} ${inputTensorHeight}`}
                scaleX={flipHorizontal}>
                {faceBoxes}
            </Svg>;
        } else {
            return null;
        }*/
    }

    handleCameraStream(images, updatePreview, gl) {
        const loop = async () => {

//blazefaceModel
            if (this.state.faceDetector != null) {
                console.log("mdfgsq")
                const imageTensor = images.next().value;
                const returnTensors = false;
                const faces = await this.state.faceDetector.estimateFaces(
                    imageTensor, returnTensors);

                this.setState({ faces });
                tf.dispose(imageTensor);
            }
            //
            // do something with tensor here
            //

            // if autorender is false you need the following two lines.
            //updatePreview();
            //gl.endFrameEXP();

            requestAnimationFrame(loop);
        }
        loop();
    }



    render() {
        // Currently expo does not support automatically determining the
        // resolution of the camera texture used. So it must be determined
        // empirically for the supported devices and preview size.

        let textureDims;
        if (Platform.OS === 'ios') {
            textureDims = {
                height: 1920,
                width: 1080,
            };
        } else {
            textureDims = {
                height: 1200,
                width: 1600,
            };
        }

        const styles = StyleSheet.create({
            container: {
                flex: 1,
            },
            camera: {
                position: 'absolute',
                left: 50,
                top: 100,
                width: 600 / 2,
                height: 800 / 2,
                zIndex: 1,
                borderWidth: 1,
                borderColor: 'black',
                borderRadius: 0,
            },
            modelResults: {
                position: 'absolute',
                left: 50,
                top: 100,
                width: 600 / 2,
                height: 800 / 2,
                zIndex: 20,
                borderWidth: 1,
                borderColor: 'black',
                borderRadius: 0,
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
        if (this.state.isLoading) {
            return (<View><Text style={{ fontSize: 30 }}>Loading</Text></View>)
        } else {
            return (<View style={styles.container}>
                <TensorCamera
                    // Standard Camera props
                    style={styles.camera}
                    type={Camera.Constants.Type.back}
                    // Tensor related props
                    cameraTextureHeight={1200}
                    cameraTextureWidth={1600}
                    resizeHeight={200}
                    resizeWidth={152}
                    resizeDepth={3}
                    onReady={this.handleCameraStream}
                    autorender={true}
                />
                <View style={styles.modelResults}>
                    {this.renderFaces()}
                </View>
            </View>)
        }
    }

}