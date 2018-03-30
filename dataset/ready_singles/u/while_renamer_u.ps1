while ($true) {
	$num = 1
	sleep 1
	$next = ".\u ($($num)).jpg"
	while((Test-Path -Path $next))
    		{
		    $num+=1   
		    $next = ".\u ($num).jpg"
		 }
	      mv .\u.jpg $next
        }	

   