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

print "----------"
print 1 + 2 + 3 - 2
print 1 + 2 * 3
print (1 + 2) * 3

print "----------"
print 1 < 0
print 1 < 2
print 1 <= 1
print 1 <= 2
print 1 <= 0
print 1 > 0
print 1 > 2
print 1 >= 1
print 1 >= 2
print 1 >= 0
print 1 = 1
print 1 = 2
print 1 <> 1
print 1 <> 2

print "----------"
print 2 > 1 and 3 > 2
print 2 > 1 + 1 and 3 > 2
print 2 > 1 + 1 and 3 > 2 or 1 > 0
print 1 > 0 or 2 > 1 + 1 and 3 > 2

print "----------"
a = 1 + 2
print a

print "----------"
if 1 > 2 then
    print 1
end if
if 2 > 1 then
    print 2
end if

print "----------"
a = 1
while a < 10
    if a mod 2 = 0 then
        print a
    end if
    a = a + 1
wend

print "----------"
function foo2()
    print 1 + 1
end function
foo2()

print "----------"
function add(a, b)
    print a + b
end function
add 1, 2

print "----------"
function showmax(a, b)
    if a > b then
        print a
    end if
    if b > a then
        print b
    end if
end function
showmax 1 + 2, 2 + 2

print "----------"
function max(a, b) 
    if a > b then
        max =  a
    end if
    max = b 
end function
m = max( 1, 2)
print m
m = max(4, 3)
print m

print "----------"
function printn(n)
    if n > 1 then
        printn n - 1
    end if
    print n
end function
printn 5

print "----------"
function aaa(n)
    if n > 1 then
        aaa = aaa(n - 1) + aaa(n - 1)
    else
        aaa = n
    end if    
end function
print aaa(5)

print "----------"
function add(a, b)
  add = a + b
end function
print add(1, add(1, 2))