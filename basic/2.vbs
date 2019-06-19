sub includeFile (fSpec)
    dim fileSys, file, fileData
    set fileSys = createObject ("Scripting.FileSystemObject")
    set file = fileSys.openTextFile (fSpec)
    fileData = file.readAll ()
    file.close
    executeGlobal fileData
    set file = nothing
    set fileSys = nothing
end sub

includeFile "common.vbs"

For i = 1 To 5
    print i
    for j = i to 5
        print j        
    next    
Next
