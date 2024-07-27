$text = "a line of text\n"; # 也可以由<STDIN>输入
chomp($text); #去掉换行符(\n)。
print $text;
chomp ($text = <STDIN>); #读入,但不含换行符
print $text
$food = <STDIN>;
$betty = chomp $food; #得到值 1
print $betty, $food;
