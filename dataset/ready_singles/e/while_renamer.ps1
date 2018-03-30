while ($true) {
	$num = 1
	sleep 1
	$next = ".\e ($($num)).jpg"
	while((Test-Path -Path $next))
    		{
		    $num+=1   
		    $next = ".\e ($num).jpg"
		 }
	      mv .\e.jpg $next
        }	

   