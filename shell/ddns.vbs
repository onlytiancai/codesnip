' name: windows 脚本版的动态域名解析
' description: 
'     1. 先在DNSPod网站上把要进行动态解析的记录的域名ID，记录ID等信息记录下来，并在下面配置好；
'     1. 在D盘根目录下创建ddns.log来存放日志，因为程序可能没有权限创建文件。
'     1. 然后把本文件加到开机启动项里就可以了，不同操作系统方法不同，请自行百度。
' author: onlytiancai@gmail.com
' last update time: 2013-01-29

Option Explicit

Dim login_email, login_password, domain_id, record_id, sub_domain, record_line

login_email = "baidu@qq.com"    'DNSPod用户名
login_password = "helloworld"   'DNSPod密码   
domain_id = 1353370             '域名ID，在DNSPod网站上获取 
record_id = 18074610            '记录ID，在DNSPod网站上获取 
sub_domain = "f"                '子域名，在DNSPod网站上获取
record_line = "默认"            '记录的线路，在DNSPod网站上获取

Function URLEncode(strURL)
    Dim I
    Dim tempStr
    For I = 1 To Len(strURL)
        If Asc(Mid(strURL, I, 1)) < 0 Then
            tempStr = "%" & Right(CStr(Hex(Asc(Mid(strURL, I, 1)))), 2)
            tempStr = "%" & Left(CStr(Hex(Asc(Mid(strURL, I, 1)))), Len(CStr(Hex(Asc(Mid(strURL, I, 1))))) - 2) & tempStr
            URLEncode = URLEncode & tempStr
        ElseIf (Asc(Mid(strURL, I, 1)) >= 65 And Asc(Mid(strURL, I, 1)) <= 90) Or (Asc(Mid(strURL, I, 1)) >= 97 And Asc(Mid(strURL, I, 1)) <= 122) Or (Asc(Mid(strURL, I, 1)) >= 48 And Asc(Mid(strURL, I, 1)) <= 57) Then
            URLEncode = URLEncode & Mid(strURL, I, 1)
        Else
            URLEncode = URLEncode & "%" & Hex(Asc(Mid(strURL, I, 1)))
        End If
    Next
End Function

Sub log(msg)
    Dim fso, ddns_log
    Set fso = CreateObject("Scripting.FileSystemObject") 
    If not fso.FileExists("d:\\ddns.log") Then
        set ddns_log = fso.createtextfile("d:\\ddns.log")
    End if
    Set ddns_log = fso.OpenTextFile("d:\\ddns.log", 8)
    ddns_log.Writeline(FormatdateTime(now, 2) + " " + FormatdateTime(now, 3) + ":" + msg)
    ddns_log.Close()
End Sub

Function InvokeApi()
    Dim objHTTP, xmlDOC, url, postData
    Set objHTTP = CreateObject("MSXML2.XMLHTTP")
    Set xmlDOC = CreateObject("MSXML.DOMDocument")
    url = "https://dnsapi.cn/Record.Ddns"
    postData = "format=xml&lang=cn" _
        + "&login_email=" + login_email  _
        + "&login_password=" + login_password _
        + "&domain_id=" + Cstr(domain_id) _
        + "&record_id=" + Cstr(record_id) _
        + "&sub_domain=" + sub_domain _
        + "&record_line=" + URLEncode(record_line)

    objHTTP.Open "POST", url, False
    objHTTP.SetRequestHeader "Content-Type", "application/x-www-form-urlencoded"
    objHTTP.Send(postData)

    Dim httpStauts, apiStatus, apiStatusCode, resultIp
    xmlDOC.load(objHTTP.responseXML)
    httpStauts = objHTTP.Status
    apiStatusCode = xmlDOC.selectSingleNode("//dnspod/status/code").text
    apiStatus = xmlDOC.selectSingleNode("//dnspod/status/message").text
    resultIp = "" 
    If apiStatusCode = "1" Then
        resultIp = xmlDOC.selectSingleNode("//dnspod/record/value").text
    End If

    InvokeApi = Cstr(httpStauts) + " " + apiStatus + " " + resultIp
End Function

While True
    log(InvokeApi())
    Wscript.Sleep 5 * 60 * 1000
Wend
