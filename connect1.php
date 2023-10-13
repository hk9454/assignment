<?php
$usern=$_POST['usern'];
$email=$_POST['email'];
$password=$_POST['password'];


//Database connection
$conn = new mysqli('localhost','root','','test2');
if($conn->connect_error){
	die('connection Failed :' .$conn->connect_error);
	else
	{
		$stmt=$conn->prepare("insert into registration(usern, email, password)
		values(?,?)");
		$stmt->bind_param("ss",$usern, $email, $password)
		$stmt->execute();
		
		$stmt->close();
		echo "registration successful ..."
		$conn->close();
	}
}

?>