 Set oWSH = CreateObject("WScript.Shell")
 vbsInterpreter = "cscript.exe"

 Call ForceConsole()

 Function printf(txt)
    WScript.StdOut.WriteLine txt
 End Function

 Function printl(txt)
    WScript.StdOut.Write txt
 End Function

 Function scanf()
    scanf = LCase(WScript.StdIn.ReadLine)
 End Function

 Function wait(n)
    WScript.Sleep Int(n * 1000)
 End Function

 Function ForceConsole()
    If InStr(LCase(WScript.FullName), vbsInterpreter) = 0 Then
        oWSH.Run vbsInterpreter & " //NoLogo " & Chr(34) & WScript.ScriptFullName & Chr(34)
        WScript.Quit
    End If
 End Function

 Function cls()
    For i = 1 To 50
        printf ""
    Next
 End Function

 printf " _____ _ _           _____         _    _____         _     _   "
 printf "|  _  |_| |_ ___ ___|     |_ _ _ _| |  |   __|___ ___|_|___| |_ "
 printf "|     | | '_| . |   |   --| | | | . |  |__   |  _|  _| | . |  _|"
 printf "|__|__|_|_,_|___|_|_|_____|_____|___|  |_____|___|_| |_|  _|_|  "
 printf "                                                       |_|     v1.0"
 printl " Enter your name:"
 MyVar = scanf
 cls
 printf "Your name is: " & MyVar
 wait(5)