/**
 * 浏览器发送通知方法
 * Author：木芒果
 * @param {Object} title 通知标题
 * @param {Object} options 可选参数body(消息体)、icon(通知显示的图标路径)、data(点击通知后跳转的URL)
 * 示例：
 * createNotify("新的消息", {
 *		body: "你有一个奖品待领取",
 *		icon: "https://www.baidu.com/favicon.ico",
 *		data: "https://www.baidu.com/"
 *	});
 */
function createNotify(title, options) {
	var PERMISSON_GRANTED = "granted";
	var PERMISSON_DENIED = "denied";
	var PERMISSON_DEFAULT = "default";
 
	// 如果用户已经允许，直接显示消息，如果不允许则提示用户授权
	if (Notification.permission === PERMISSON_GRANTED) {
		notify(title, options);
	} else {
		Notification.requestPermission(function(res) {
			if (res === PERMISSON_GRANTED) {
				notify(title, options);
			}
		});
	}
 
	// 显示提示消息
	function notify($title, $options) {
		var notification = new Notification($title, $options);
		console.log(notification);
		notification.onshow = function(event) {
			console.log("show : ", event);
		};
		notification.onclose = function(event) {
			console.log("close : ", event);
		};
		notification.onclick = function(event) {
			console.log("click : ", event);
			// 当点击事件触发，打开指定的url
			window.open(event.target.data)
			notification.close();
		};
	}
}
/* 依次打印
 * show:   Event Object(事件对象)，事件的type为"show"
 * click:  Event Object(事件对象)，事件的type为"click"。点击消息后消息被关闭，跳到close事件。
 * close:  Event Object(事件对象)，事件的type为"close"
 */
