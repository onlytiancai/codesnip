/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 */

import React, {
    AppRegistry,
    Component,
    Image,
    StyleSheet,
    Text,
    View,
    ListView,
    Platform,
    TouchableHighlight,
    TouchableNativeFeedback,
    Linking,
} from 'react-native';

var Dimensions = require('Dimensions');
var windowWidth = Dimensions.get('window').width;

var baidu_api_url = 'http://image.baidu.com/data/imgs?col=%E7%BE%8E%E5%A5%B3&tag=%E5%B0%8F%E6%B8%85%E6%96%B0&sort=0&rn=10&p=channel&from=1&pn=';

class AwesomeProject extends Component {
    constructor(props) {
        super(props);

        this.state = {
            dataSource: new ListView.DataSource({
                rowHasChanged: (row1, row2) => row1 !== row2,
            }),
            loaded: false,
        };

        this.pn = 0;

    }

    componentDidMount() {
        this.fetchData();
    }

    fetchData() {
        var url = baidu_api_url + this.pn;
        console.log(url);
        this.pn += 1;

        this.setState({
            loaded: false,
        });

        fetch(url)
            .then((response) => response.json())
            .then((responseData) => {
                this.setState({
                    dataSource: this.state.dataSource.cloneWithRows(responseData.imgs),
                    loaded: true,
                });
            })
            .catch(err => console.log(err))
            .done();
    }

    render() {
        var TouchableElement = TouchableHighlight;
        if (Platform.OS === 'android') {
            TouchableElement = TouchableNativeFeedback;
        }

        if (!this.state.loaded) {
            return this.renderLoadingView();
        }

        return (
            <View style={styles.container}>
                <Text style={styles.title}>Next Girl</Text>
                <ListView
                    dataSource={this.state.dataSource}
                    renderRow={this.renderImg}
                    style={styles.listView}
                    />
                <TouchableElement
                    onPress={this.fetchData.bind(this) }>
                    <View>
                        <Text style={styles.load_more} >Load more girls</Text>
                    </View>
                </TouchableElement >
            </View >
        );
    }

    renderLoadingView() {
        return (
            <View style={styles.container}>
                <Text style={styles.title}>Next Girl</Text>
                <View style={styles.loading_container}>
                    <Text style={styles.loading}>
                        Loading girls...
                    </Text>
                </View>
            </View>
        );
    }

    renderImg(img) {
        let width = windowWidth > img.imageWidth ? img.imageWidth : windowWidth;
        let height = width * (img.imageHeight / img.imageWidth);
        return (
            <View style={styles.container}>
                <TouchableHighlight onPress={ () => Linking.openURL(img.imageUrl) }>
                    <Image
                        source={{ uri: img.imageUrl }}
                        style={{
                            width: width,
                            height: height,
                            backgroundColor: 'transparent',
                        }}
                        />
                </TouchableHighlight>
                <Text>{img.desc}</Text>
                <Text>{img.date}</Text>
            </View>
        );
    }
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#7fffd4',
    },
    listView: {
    },
    title: {
        backgroundColor: '#ff7f50',
        color: '#ffffff',
        fontSize: 22,
        fontWeight: 'bold',
        textAlign: 'center',
        fontFamily: 'Cochin',
    },
    loading_container: {
        flex: 1,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#7fffd4',
    },
    loading: {
        fontFamily: 'Cochin',
    },
    load_more: {
        color: '#ffffff',
        backgroundColor: '#ff7f50',
        textAlign: 'center',
    }
});

AppRegistry.registerComponent('AwesomeProject', () => AwesomeProject);
