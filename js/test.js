
function printspaces(){
	let result = document.getElementById("result");	// must have an element node
	let text_node = document.createTextNode();		// to create a text node to add into HTML		guess_space.appendChild(text_node);			// append text node to element in HTML
	result.appendChild(text_node);			// append text node to element in HTML

}

let dosomething = function(){
	output=document.getElementById("result");
	printspaces(;)
};

let checkLetter = function(){
	// let f = document.guess_form;		// form from HTML
	// let b = f.elements["input_letter"];	// the "input_letter" element from the form
	// let letter = b.value; 				// the letter provided by the user into "input_letter" element
	// letter = letter.toUpperCase();
	// let found = false;
	// // here, we check if the user's guessed letter is a letter in the word (chosen)
	// for (let i = 0; i < chosen.length; i++){
	// 	if(chosen[i] === letter){
	// 		found = true;
	// 		spaces[i] = letter + " ";	// replace spaces[i] with the letter found
	// 	}
	// }
	// b.value = "";		// empty out text input box for next round
	
	// deletes the guessfield and replaces it with the new one
	let output= document.getElementById("output");
	output.innerHTML=""; 
	printspaces();
	

}

printspaces();