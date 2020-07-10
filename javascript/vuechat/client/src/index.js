import _ from 'lodash';
import './style.css';
import Logo from './logo.jpg';

function component() {
  var element = document.createElement('div');

	// Lodash，现在由此脚本导入
  element.innerHTML = _.join(['Hello', 'webpack'], ' ');
	element.classList.add('hello');


	var myIcon = new Image();
	myIcon.src = Logo;

	element.appendChild(myIcon);

  return element;
}

document.body.appendChild(component());
