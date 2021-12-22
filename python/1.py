a= "'not h i'"
b=a.replace('\'',"").replace(' and ','" AND "').replace(' or ','" OR "').replace(' not ','" NOT "')
c= '\'"' + b + '"\''
print(c)
