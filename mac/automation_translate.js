// 工作流出收到：文本，位于：任何应用程序
function run(input, parameters) {
    var theText = input.toString();
    var app = Application.currentApplication();
    app.includeStandardAdditions = true;
	
	var apiKey = app.doShellScript('source ~/.zshrc; echo $GOOGLE_API_KEY').trim();
	var targetLanguage = "zh";
    var apiUrl = "https://translation.googleapis.com/language/translate/v2?key=" + apiKey;
    var jsonData = JSON.stringify({ q: theText, target: targetLanguage });
	var proxy = "http://127.0.0.1:10808";
	
    var curlCommand = 'curl -s -X POST "' + apiUrl + '" ' +
                      '-H "Content-Type: application/json" ' +
                      '-d ' + "'" + jsonData + "' " +
					  '-x ' + proxy + ' ' + 
                      '--connect-timeout 3 ' + 
                      '--max-time 5'; 

	var result, exitCode;
	try {
    	result = app.doShellScript(curlCommand, {administratorPrivileges: false});
    	exitCode = 0;
	} catch (e) {
    	result = e.message;
    	exitCode = e.number;
	}

	if (exitCode !== 0) {
    	app.displayDialog("curl 执行失败，退出码：" + exitCode + "\n输出：" + result, {buttons: ["OK"]});
	} else {
    	// 正常解析 JSON
    	var translation;
    	try {
        	var parsed = JSON.parse(result);
        	translation = parsed.data.translations[0].translatedText;
    	} catch (e) {
        	translation = "解析失败: " + e.message + "\n响应: " + result;
    	}
    	app.displayDialog("原文："+ theText +"\n\n翻译结果: " + translation, { buttons: ["OK"], defaultButton: "OK" });
	}
}
