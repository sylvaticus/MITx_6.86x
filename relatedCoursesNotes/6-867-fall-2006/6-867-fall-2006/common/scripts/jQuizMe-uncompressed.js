// jQuizMe 2.2 by Larry Battle.
//Please give me feedback at blarry@bateru.com
//Copyright (c) 2010 Larry Battle 12/30/2010
//Dual licensed under the MIT and GPL licenses.
//http://www.opensource.org/licenses/mit-license.php
//http://www.gnu.org/licenses/gpl.html
// quizData =[ { ques: "", ans : ""},  {ques: "", ans : ""}, etc ];
// quiz( quizData, options ) # wordsList is an arrary of objects with the following format

(function($){	
	var _jQuizMeLayOut = $("<div/>").addClass( "quiz-el").append( 
		$("<div/>").addClass( "q-header q-innerArea").append(
			$("<div/>").addClass( "q-counter"),
			$("<div/>").addClass( "q-title")
		),
		$("<div/>").addClass( "q-help q-innerArea").append( 
			$("<div/>").addClass( "q-help-menu" ).append(
				$( "<span/>" ).addClass( "q-quit-area" ).append(
					$( "<input type ='button'/>" ).addClass( "q-quit-btn" ),
					$( "<input type='button'/>" ).addClass( "q-quitYes-btn q-quit-confirm" ).hide(),
					$( "<input type='button'/>" ).addClass( "q-quitNo-btn q-quit-confirm" ).hide()
				),
				$("<input type='button'/>").addClass( "q-help-btn" ),									
				$( "<span/>" ).addClass( "q-timer-area" ),
				$("<div/>").addClass( "q-help-info" ).hide()
			)
		),
		$("<div/>").addClass( "q-review-menu q-innerArea").append( 
			$("<input type='button'/>").addClass( "q-details-btn q-reviewBar-btns" ),
			$("<input type='button'/>").addClass( "q-review-btn q-reviewBar-btns" ),
			$("<div/>").addClass( "q-reviewBar q-innerArea").append(
				$("<input type='button'/>").attr({ "class": "q-leftArrow q-review-arrows", "value": "<-" }),
				$("<input type='button'/>").attr({ "class": "q-rightArrow q-review-arrows", "value": "->" }),
				$( "<span/>" ).addClass( "q-review-nav" ).text( 'Reviewing question ' ).append(
					$("<select/>").addClass( "q-review-index-all q-review-index"),
					$("<select/>").addClass( "q-review-index-missed q-review-index").hide(),
					$("<span/>").addClass( "q-missMarker" ).show(),
					$("<input type='checkbox'/>").addClass( "q-showOnlyMissed-btn")
				)
			).hide()
		),
		$("<div/>").addClass( "q-intro q-innerArea" ).append(
			$( '<p/>' ).addClass( "q-intro-info" ),
			$( '<input type="button"/>' ).addClass( "q-begin-btn" )
		).hide(),
		$("<div/>").addClass( "q-prob q-innerArea" ).append( 
			$("<div/>").addClass( "q-ques q-probArea"),
			$("<div/>").addClass( "q-ans q-probArea").append(
					$( "<div/>" ).addClass( "q-ansSel" ),
					$( "<input type='button'/>" ).addClass( "q-check-btn" ),
					$( "<input type='button'/>" ).addClass( "q-next-btn" )
				),
			$("<div/>").addClass( "q-result q-probArea" ).hide() 
		),
		$("<div/>").addClass( "q-gameOver q-innerArea" ).append(
			$("<div/>").addClass( "q-stat q-probArea").append(
					$("<span/>").addClass( "q-statTotal q-center"),
					$("<hr/>"),
					$("<blockquote/>").addClass( "q-statDetails"),
					$("<div/>").addClass( "q-extraStat" )	
			),
			$("<div/>").addClass( "q-options q-probArea q-center").append( 
				$("<input type='button'/>").addClass( "q-restart-btn" ),
				$("<input type='button'/>").addClass( "q-del-btn" ) 
			)
		).hide()
	);
	// Note: areArraysSame will fail when there is an object within the array.
	var areArraysSame = function( mainArr, testArr) {
		if ( mainArr.length != testArr.length ){ return false; }
		var i = mainArr.length;	
		
		while( i-- ) {
			if ( ( isArray( mainArr[i] ) && !areArraysSame( mainArr[i], testArr[i] ) ) || mainArr[i] !== testArr[i] ) { 
				return false;
			}
		}
		return true;
	},
	getStrWithSeeableHTML = function( str, allow ) {
		if( !allow || typeof str !== "string" ){ return str; }
		return str.replace( /</g, '&#60;' );
	},
	getStrArrOfSeeableHTML = function( arr, allow ){
		if( !allow ){ return arr; }
		if( isArray( arr ) ){
			var i = arr.length;
			while( i-- ){
				arr[ i ] = getStrWithSeeableHTML( arr[ i ], true );
			}
		}
		return arr;
	},
	getNumRange = function( i ){
		var arr = [];
		while( i-- ){
			arr[ i ] = i;
		}
		return arr;
	},
	getObjProps = function( obj ){
		var keys = [];
		for( var prop in obj ){
			if( obj.hasOwnProperty( prop ) ){
				keys[ keys.length ] = prop;
			}
		}	
		return keys;
	},
	rNum = function( len ){
		return Math.floor( Math.random() * len );
	},
	rBool = function(){
		return ( (rNum( 100 ) % 2) != 1);
	},
	makeArrayRandom = function( arr ){
		var j, x, i = arr.length;
		while( i ){
			j = parseInt(Math.random() * i, 10);
			x = arr[--i]; 
			arr[i] = arr[j];
			arr[j] = x;
		}
		return arr;
	},	
	getArrayOfSwappedArrayElements = function( arr, iArr ){
		if( !isArray( arr ) || !isArray( iArr ) || arr.length !== iArr.length ){
			return -1;
		}
		var x = [], i = arr.length;
		while( i-- ){
			x[ i ] = arr[ iArr[ i ] ];
		}
		return x;
	},
	isArray = function( obj ){
		return Object.prototype.toString.call( obj ) == "[object Array]";
	},
	isMSIE = window.navigator.userAgent.match(/MSIE \d/),
	isBadMSIE = ( isMSIE && isMSIE[0].match(/\d/)[0] < 9 ),
	_lang = {
			ans:{	// _lang.ans are shown during the display of .q-result, the answer result.
				corrAns: "<strong>Correct answer(s):</strong>",
				praise: '<strong>Great Job. Right!</strong>',
				retry: 'Incorrect.<br/>Please try another answer.',
				whyAns: "<strong>Explanation:</strong>",
				yourAns: "<strong>Your Answer:</strong>"
			},
			btn:{	// [ "text", "title" ]
				begin: [ "Begin Quiz" ],
				check: [ "Check", "Check your answer" ],
				del: [ "Delete", "Delete quiz" ],
				help: [ "Help", 'Click for help'],
				next: [ "Next", "Next question" ],
				restart: [ "Restart", "Restart the quit over" ],
				details: [ "Score Report", "View score report" ],
				review: [ "Review Answers", "Review Questions" ],
				showOnlyMissed: [ " *Click to show only missed questions " ],
				quit: [ "Quit", "Quit quiz" ],
				quitNo: [ "->", "Go Back" ],
				quitYes: [ '', "Yes, quit" ]
			},
			err:{
				ansInfoNotDefined: "Property ansInfo must be a string and be defined when ansSelInfo is defined.",
				badAnsSelInfoLen: "Property ansSel and ansSelInfo must be the same type and have the same length if an array.",
				badQType: "Invalid quiz type.",
				badKey: " is an invalid key value.",
				error: "Error",	
				noQType: "No quizTypes.",
				noQues: "Quiz has no questions.",
				notArr: "Must be an array.",
				notObj: "Invalid quiz data structure. Must be an array or object."
			},
			stats:{
				right: "Right",
				rate: "Rate",
				score: "Score",
				wrong: "Wrong",
				total: "Total",
				tried: "Tried"
			},
			quiz:{
				tfEqual: " = ",
				tfEnd : "?",
				tfFalse: "False",
				tfTrue: "True"
			}
	},
	setLangBtnTxt = function(langBtn, layout) {
		var btnCls = [ "begin", "check", "del", "help", "next", "restart", "quit", "quitYes", "quitNo", "review", "details", "showOnlyMissed" ];
		var i = btnCls.length,
		el; 
		// !! Replace showOnlyMissed with an input button value="show only wrong" || "show right"
		// !! Are delete showOnlyMisssed and have * for the missed questions.
		if (!langBtn.quitYes[0]) {
			langBtn.quitYes[0] = langBtn.quit[0] + "?";
		}
		$(".q-missMarker", layout).html(langBtn.showOnlyMissed[0]);
		langBtn.showOnlyMissed[0] = '';
		while (i--) {
			el = [".q-", btnCls[i], "-btn"].join('');
			$(el, layout).attr("value", langBtn[btnCls[i]][0]);
			$(el, layout).attr("title", langBtn[btnCls[i]][1]);
		}
		return layout;
	},
	_settings = {
		addToEnd: "",  // This is attached to fill in the blank quiz types.
		activeClass: "q-ol-active", // Used on multiple choice, (multiOl), quiz types.
		allQuizType: '', // This sets all questions to this quiz type.
		allRandom: false, // Randomizes all the questions, regardless of quiz type section.
		alwaysShowAnsInfo: true, // Shows answer information when correct or incorrect answer are received.
		disableRestart: false, // Hide or show the restrart button on gameOver.
		disableDelete: false, // Hide or show the delete button on gameOver.
		enableRetry: false, // Allows the user to repeat a problem before advancing to next question. 
		fxType: 0, // animateType [ show/hide, fadeToggle, slideToggle, weight&heightToggle ];
		fxCode: false, //If a function, then this is used for animation. Please refer to animateType[] for examples.
		fxSpeed: "normal", // "fast", "slow", "normal" or numbers. 
		help: 'None', // Provide help text/html if needed.
		hoverClass: "q-ol-hover", // Used on multiple choice, (multiOl), quiz types.
		intro: '', // Provide an text/html intro.
		multiLen : 3, // Set the number of multiple choice choices for quizType multi & multiList.
		numOfQuizQues: 0, // Sets the number of questions asked. Must be between 0 and the total questions.
		performErrorChecking : true, // Check all the quiz formats.
		random: false, // Randomizes all the questions in each quiz type section.
		review: true, // Allows for review of questions at gameOver.
		showFeedback: true, // Show the answers after each question.
		showAnsInfo: true, // If provided, show the answers information after each question.
		showHTML: false, //This will show the HTML, by converting to unicode, rather than render it.
		showWrongAns: false,  // If answer is wrong, then show the user's wrong answer after each question.
		statusUpdate: false, // Sends a status Update. Refer to function sendStatus.
		quizType: "fillInTheBlank", // This is only need if you send in an array and not a object with a defined quiz type.
		title: 'jQuizMe' // title displayed for quiz.
	};
		
	$.fn.jQuizMe = function( wordList, options, userLang ){
	
		var settings = $.extend({},_settings, options),
			lang = $.extend(true, {},_lang, userLang),
			layout = setLangBtnTxt( lang.btn, _jQuizMeLayOut.clone(true));

		return this.each( function(){
			// currQuiz is the output file.(this is what the user sees). $( el, currQuiz) must be used when accessing elements.
			// Hide currQuiz until it's ready to be displayed.
			var currQuiz = layout.clone(true).hide(), currIndex = 0, stickyEl = this, quit = false, totalQuesLen = 0,
			// The q object is where the quiz data, that was made before the quiz starts, is stored here.
			// Please beware that all the questions are stored linearly. But you can find a quiz type by using index[Max|Min].
			q = {
				// Each index is a question. [ question1, question2, question3,... ]. But not for indexMax|Min and props.
				ans: [], // Answers.
				ansSel: [], // Answer selections, the elements for inputting an answer. Ex. input text box, or a select tag.
				ansSelInfo: [], // Used to show specific feedback for ansSel choices.
				ansInfo: [], // Stores the answer information if provided.
				retryCount: [], // Stores the number of retries that a user can have.
				indexMax: [], // The stopping point index for a quiz type inside the q. ans, ansSel, ansInfo and ques.
				indexMin: [], // The starting point index.
				prop:[], // Stores the quiz types names in order. Used for creation of each.
				ques: [] // Questions.
			},
			stats = { 
				indexSelected: [], // indexSelected is only used when ansSelInfo is used.
				problem: {},
				numOfRight: 0, 
				numOfWrong : 0,
				quesTried: 0,
				totalQues: 0,
				accur : function(){
					var x = Math.round( this.numOfRight / this.quesTried * 100 );
					return ( this.quesTried ) ? x : 0;
				},
				accurTxt : function(){
					return [ this.numOfRight, "/", this.quesTried," = ", this.accur(), "%"].join('');
				},
				perc : function(){ 
					var x = Math.round( this.numOfRight / this.totalQues * 100 );
					return ( this.totalQues ) ? x : 0; 
				},
				percTxt : function(){
					return [ this.numOfRight, "/", this.numOfQues, " = ", this.perc(), "%"].join('');
				},
				reset : function(){ 
					this.totalQues = totalQuesLen;
					this.quesTried = 0;
					this.numOfRight = 0;
					this.numOfWrong = 0;
					this.problem = {};
					this.indexSelected = [];
				}
			},
			// rAns(): Returns a random answer from either all the questions, or a curtain quiz type.
			// iProp -> index of the desired quiz type.
			rAns = function( iProp, fromAllQues ){
				var p = ( fromAllQues ) ? q.prop[ rNum( q.prop.length ) ] : q.prop[ iProp ],
					ques = wordList[ p ][ rNum( wordList[ p ].length ) ];

				return ques.ans;
			},
			disableMenuBar = function(){
				$( ".q-quit-btn, .q-help-btn", currQuiz ).attr( "disabled", true );
				$( ".q-help-info", currQuiz ).hide();
			},
			haveCorrAnsWithinRetries = function( isFlashCard ){			
				return ( ((getProblemProp( "amountTried" ) - 1) <= q.retryCount[ currIndex ]) && checkAns( isFlashCard ));
			},
			letUserAnswerAgain = function( isFlashCard ){
				return (settings.enableRetry && getProblemProp( "amountTried" ) < q.retryCount[ currIndex ] && !checkAns( isFlashCard ));
			},
			setCheckBtnToBeBlockedUntilAnsSelIsClicked = function(){
				$( ".q-check-btn", currQuiz ).attr( "disabled", true );
				$( ".userInputArea", currQuiz ).one( "click", function(){
					$( ".q-check-btn", currQuiz ).attr( "disabled", function(){
						return !$( ".q-next-btn", currQuiz ).attr( "disabled" );
					});
				});
			},
			displayRetryMsg = function(){
				var show = lang.ans.retry;
				if( settings.showAnsInfo ){
					show += getAnsInfoForDisplay();
				}
				displayFeedback( show );
			},
			// setCheckAndNextBtn(): Sets the check button to toggle from "check" to "next". Or just displays "next".
			// "check" shows the answer results, but "next" just changes the question.
			setCheckAndNextBtn = function(){
				var isFlashCard = '';
				
				$( ".q-check-btn", currQuiz ).attr( "disabled", true )
					.click( function( e ){
						isFlashCard = $( ".userInputArea", currQuiz ).triggerHandler( "getUserAns" );
						
						getStatsProblem().amountTried++;
						if( getProblemProp( "amountTried" ) == 1){
							stats.quesTried++;
						}
						if( !letUserAnswerAgain( isFlashCard ) ){
							stats[ ( checkAns( isFlashCard ) || isFlashCard ) ? "numOfRight" : "numOfWrong" ]++;
							stats.problem[ currIndex ].isCorrect = checkAns( isFlashCard );
							displayAnsResult( isFlashCard );
							$( ".q-next-btn", currQuiz ).attr( "disabled", false );
							$( e.target ).attr( "disabled", true );
						}
						else{
							setCheckBtnToBeBlockedUntilAnsSelIsClicked();
							displayRetryMsg();
						}
						e.preventDefault();
					}
				);
				$(".q-next-btn", currQuiz ).click( function(e){
					nextMove();
					$( e.target ).attr( "disabled", true );
					e.preventDefault();
				});
			},
			// setQuitBtn(): Provides a quit confirm action.
			setQuitBtn = function(){
				$( ".q-quitYes-btn", currQuiz ).click( function(){ 
					quit = true;
					nextMove();
				});
				$( ".q-quit-confirm", currQuiz ).click( function(){ 
					$( ".q-quit-confirm", currQuiz ).hide(); //.q-quit-confirm calls both the quitYes and quitNo btns.
					$( ".q-quit-btn", currQuiz ).css("display", "inline");
				});
				$( ".q-quit-btn", currQuiz ).click( function( e ){ 
					$( e.target ).hide();
					$( ".q-quit-confirm", currQuiz ).css("display", "inline");
				}); 
			},
			setTotalQuesLen = function(){	
				var i = q.prop.length, len = 0, numOfQQ = settings.numOfQuizQues;
				
				while( i-- ){
					len += wordList[ q.prop[ i ] ].length; 
				}
				totalQuesLen = ( !numOfQQ || isNaN( numOfQQ ) || numOfQQ > len ) ? len : numOfQQ;
				stats.numOfQues = totalQuesLen;
			},
			// setQuizTypePos(): This stores the position of the beginning and end index of each quiz type.
			// This allows for each quiz type to be pinpointed to in an array.
			setQuizTypePos = function(){
				var i = 0, sum = 0, len = 0;
				
				while( q.prop[ i ] ){ 
					len = wordList[ q.prop[ i ] ].length;
					sum += len; 
					q.indexMax[ i ] = sum - 1;
					q.indexMin[ i ] = sum - len;
					i++;
				}
			},
			// createQuizEl(): Creates new quiz element and sets default button behaviors.
			createQuizEl = function(){
				$( ".q-help-btn", currQuiz).toggle( 
					function(){ animateThis( $( ".q-help-info", currQuiz), 1 ); },
					function(){ animateThis( $( ".q-help-info", currQuiz), 0 ); }
				);
				setTotalQuesLen();
				setQuizTypePos();
				setQuitBtn();
				setCheckAndNextBtn();
				if( settings.review ){ setReviewMenu(); }
				$( stickyEl ).append( currQuiz );	
			},
			// cont(): Checks whether the questions are done or if quit was called.
			cont = function(){ 
				var result = ( !quit && ( currIndex < totalQuesLen ) ); 
				if( !result ){ quit = true; }
				return result;
			},
			// animateType[]: Stores animation styles to animate an element.
			animateType = [
				function( el, on ){ el[ ( on ) ? "show" : "hide" ]( 0 );},
				function( el, on, spd ){ el[ ( on ) ? "fadeIn" : "fadeOut" ]( spd );},
				function( el, on, spd ){ el[ ( on ) ? "slideDown" : "slideUp" ]( spd );},
				function( el, on, spd ){ 
					el.animate( { "height": "toggle", "width": "toggle" }, spd ); 
				}
			],
			// animateThis(): Calls animations on an element and/or push a callback function.
			animateThis = function( el, isShown, func ){
				if( isShown > -1 ){ 
					animateType[ settings.fxType ]( el, isShown, settings.fxSpeed ); 
				}
				if( func ){ 
					el.queue( function(){ func(); $(el).dequeue();});
				}
				if( isBadMSIE && (isShown > 0) ){
					$(el).animate( { opacity: 1} ).css( "display", "block" );
				}
			},
			// setReviewNav(): Adds events and functions to change the review questions being viewed.
			// changeCurrIndex() is main point of focus.
			setReviewNav = function(){
				var qReviewOpt = ".q-review-index:visible option:", 
					qReviewOptVal = qReviewOpt + "selected",
					changeCurrIndex = function( i ){
						var max = $( qReviewOpt + "last", currQuiz).val(),
							min = $( qReviewOpt + "first", currQuiz).val(),
							isIndexValid = ( i <= max && i >= min );
							
						if( isIndexValid ){
							$( ".q-review-index", currQuiz).val( i );
							currIndex = i;
							changeProb(true);
							displayAnsResult();
							$( ".q-reviewBar > :disabled", currQuiz).attr( "disabled", false );
						}
						// if a limit was reached. Disable the corresponding arrow.
						if( i == max || i == min ){
							$( ".q-" + (( i == min )?"left":"right") + "Arrow", currQuiz ).attr( "disabled", true );
						}
					};
				$( ".q-review-index", currQuiz).change(function(e){
					changeCurrIndex( parseInt( $(e.target).val(), 10) );
				});
				// Note: The prev||next values of the options tag are used because the list might not be in numerical order.
				$( ".q-review-arrows", currQuiz).click(function( e ){
					var isLeft = !!e.target.className.match( /left/ ), 
						reviewVal = $( qReviewOptVal, currQuiz)[ ( isLeft ) ? "prev":"next" ]().val();
					
					changeCurrIndex( parseInt( reviewVal, 10) ); 
				});
			},
			// setReviewMenu(): Sets the events to change the view when the "review" or "details" button is clicked.
			setReviewMenu = function(){
				var qG = $( ".q-gameOver", currQuiz ), qP = $( ".q-prob", currQuiz );
				
				$( ".q-reviewBar-btns", currQuiz ).click( function( e ){
					var isDetailEl = ( e.target.className.search(/details/) > -1),
						showEl = ( isDetailEl ) ? qG : qP,
						hideEl = ( !isDetailEl ) ? qG : qP;
					
					animateThis( $( ".q-reviewBar", currQuiz), !isDetailEl );
					animateThis( hideEl, 0, function(){
						$( ".q-details-btn", currQuiz ).attr( "disabled", isDetailEl );
						$( ".q-review-btn", currQuiz ).attr( "disabled", !isDetailEl );
						animateThis( showEl, 1 );
					});
				});
				$( ".q-showOnlyMissed-btn", currQuiz ).change( function(){
					var ckd = $(this).is(":checked");
					if( ckd ){
						$( ".q-review-index-missed", currQuiz ).fadeIn().css( "display", "inline" );
						$( ".q-review-index-all", currQuiz ).hide();
					}
					else{
						$( ".q-review-index-missed", currQuiz ).hide();
						$( ".q-review-index-all", currQuiz ).fadeIn().css( "display", "inline" );
					}
					$( ".q-review-index:visible", currQuiz ).trigger( "change" ).css( "display", "inline" );
				});
				setReviewNav();
			},
			// updateReviewIndex(): Creates the review's select tag html for all and wrong answers.
			updateReviewIndex = function(){
				var allMissed = '', optHtml = '', markAsWrong = '', eachOpt = '',
					i = 0, totalQues = stats.totalQues;
				
				while( totalQues - i ){
					markAsWrong = ( getProblemProp( "isCorrect", i ) ) ? "" : " *";
					eachOpt = "<option value=" + i + ">";
					eachOpt += (i+1);
					eachOpt += markAsWrong;
					eachOpt += "</option>";
					optHtml += eachOpt;
					allMissed += (markAsWrong) ? eachOpt : "";
					i++;
				}
				$( ".q-review-index-all", currQuiz).html( optHtml );
				$( ".q-review-index-missed", currQuiz).html( allMissed );
			},
			// disableGameOverButtons(): If requested, on gameOver hide the restart and/or delete buttons.
			disableGameOverButtons = function(){
				if( settings.disableRestart ){ $( ".q-restart-btn", currQuiz).hide(); }
				if( settings.disableDelete ){ $( ".q-del-btn", currQuiz).hide(); }
				if( settings.disableRestart && settings.disableDelete ){
					$( ".q-options", currQuiz ).hide();
				}
			},
			getUserStatDetailsForDisplay = function(){
				return [
						(lang.stats.right + ": " + stats.numOfRight), 
						(lang.stats.wrong + ": " + stats.numOfWrong),
						(lang.stats.tried + ": " + stats.quesTried), 
						(lang.stats.rate + ": " + stats.accurTxt()), 
						(lang.stats.total + ": " + stats.percTxt())
					].join('<br/>');
			},
			setupReview = function(){
				updateReviewIndex();
				$( ".q-review-btn", currQuiz ).one( "click", function(){
					$( ".q-review-index", currQuiz).trigger( "change" );
					displayAnsResult(); 
					// This forces the answer info to be displayed for first question.
					if( totalQuesLen == 1 ){
						$( ".q-rightArrow, .q-showOnlyMissed-btn", currQuiz ).attr( "disabled", true );
					}						
				});
				$( ".q-details-btn", currQuiz).attr( "disabled", true );
				$( ".q-review-menu", currQuiz).show();
			},
			// gameOver(): Creates a performance list, set's the events for restart and delete, hides unneed elements,
			// then enables the review-bar.
			gameOver = function(){ 
				disableGameOverButtons();
				$( ".q-statTotal", currQuiz).html( lang.stats.score + ": " + stats.perc() + "%" );
				$( ".q-statDetails", currQuiz).html( getUserStatDetailsForDisplay() );
				$( ".q-restart-btn", currQuiz).one( "click", reStartQuiz );			
				$( ".q-del-btn", currQuiz).one( "click", deleteQuiz );
				$( ".q-help, .q-check-btn, .q-prob, .q-intro, .q-ans", currQuiz).hide();
				if( settings.review ){
					setupReview();
				}
				animateThis( $( ".q-gameOver", currQuiz), 1, sendStatus );
			},
			// changeProb(): Changes to next problem and updates status.
			changeProb = function( isReview ){
				var qAS = q.ansSel;
				$( ".q-counter", currQuiz).text( 'Question ' + (currIndex + 1) + ' of ' + totalQuesLen );
				$( ".q-ques", currQuiz).html( q.ques[ currIndex ] );
				if( !isReview ){ 
					$( ".q-ansSel", currQuiz).html( 
					// Checks for an index pointer; used to clone an identical q.ansSel. This was done to speed up buildQuizType.
						qAS[ isNaN( qAS[currIndex] ) ? currIndex : qAS[currIndex] ].clone(true)
					);
					setQResultDisplay( 0 );
					$( ".q-result", currQuiz ).hide();
					$( ".focusThis", currQuiz ).focus();
					setTimeout( function(){
						$( ".clickThis", currQuiz ).trigger("click"); 
						// IE BUG FIX. The list-style-type won't get rendered until hovered. 
						// Resetting the property will forces it to render. Runs slow though.
						var qOlLiEl = $( ".q-ol-li",  currQuiz );
						if( qOlLiEl.length && isBadMSIE ){
							qOlLiEl.css( "list-style-type", qOlLiEl.css( "list-style-type" ) );
						}
					}, 15);
				}
			},
			setHelpBtn = function(){
				if (settings.help) {
					$(".q-help-info", currQuiz).html(settings.help);
				} else {
					$(".q-help-btn", currQuiz).attr("disabled", true);
				}
			},
			// changeQuizInfo(): Changes the main information for the quiz.
			changeQuizInfo = function(){
				setToBeginningView();
				$( ".q-title", currQuiz).html( settings.title );
				setHelpBtn();
				if( settings.intro ){
					$( ".q-prob", currQuiz ).hide();
					$( ".q-intro-info", currQuiz ).html( settings.intro );
					$( ".q-intro", currQuiz ).show();
					$( ".q-begin-btn", currQuiz ).unbind().one( "click", function(){
						animateThis( $( ".q-intro", currQuiz ), 0, function(){ 
							animateThis( $( ".q-prob", currQuiz ), 1, sendStatus );
						});
					});
				}			
			},
			goToNextProb = function( i ){
				if( i && i < totalQuesLen && (currIndex + 1) < i ){
					currIndex = i;
				}
				changeProb();
			},
			sendStatus = function(){
				if( !settings.statusUpdate ){ 
					return;
				}
				var quizInfo = { 
					"currIndex": currIndex,
					"problem": stats.problem,
					"numOfRight":stats.numOfRight,
					"numOfWrong":stats.numOfWrong,
					"score": stats.perc(),
					"total": stats.totalQues,
					"hasQuit": quit,
					"deleteQuiz": deleteQuiz,
					"quitQuiz": ( !quit ) ? quitQuiz : function(){},
					"nextQuestion": ( !quit ) ? goToNextProb : function(){}
				};
					
				settings.statusUpdate( quizInfo, currQuiz );
			},
			// nextMove(): Decides to change the problem or to quit.
			nextMove = function(){
				currIndex++; // currIndex++ must be first, to find out if there are other questions.
				var nextFunc = (cont() ? changeProb : quitQuiz),
					qEl = $( ".q-prob", currQuiz);
				
				if( !quit ){ animateThis( qEl, 0 ); }
				animateThis( qEl, -1, nextFunc ); //Push the function to the stack, so it's called in between the animations.
				if( !quit ){ animateThis( qEl, 1, sendStatus ); }
			},
			// getAns(): Is used to check the answers. 
			getAns = function( i ){ 
				return q.ans[ i || currIndex ]; 
			},
			getStatsProblem = function( i ){
				i = i || currIndex;
				if( !stats.problem[ i ] ){
					stats.problem[ i ] = { "amountTried": 0, "isCorrect":null, "userAnswer": undefined };
				}
				return stats.problem[ i ];
			},
			getProblemProp = function( prop, i ){
				return getStatsProblem( i )[ prop ];
			},
			setUserAnswer = function( value, i ){
				getStatsProblem( i ).userAnswer = value;
			},
			// checkAns(): Changes the user's answer during the quiz and updates the stats.
			checkAns = function( isFlashCard ){
				var ans = getAns(), isAnsCorr = false, userAns = getProblemProp( "userAnswer" );
				
				if( userAns === undefined ){ return false; }
				if( typeof(ans) === "string"){
					ans = $.trim( ans );
				}
				if( isArray( ans ) ){
					// If one element of the array matches the user's input, then it's correct.
					// This enables multiple answers to be possible.
					isAnsCorr = ( userAns == ans.toString() || ($.inArray( userAns, ans ) + 1));
				}
				else{
					isAnsCorr = ( userAns == ans || userAns.toString().toLowerCase() == ans.toString().toLowerCase() );
				}
				return isAnsCorr;
			},
			getUserAnsForDisplay = function( toUni ){
				var ans = lang.ans.yourAns + "<br/>";

				return ( !getProblemProp( "userAnswer" ) ) ? "<del>" + ans +"</del>" :
							ans + getStrWithSeeableHTML( getProblemProp( "userAnswer" ), toUni );
			},
			getAnsInfoForDisplay = function(){
				if( !q.ansInfo[ currIndex ] ){
					return "";
				}
				if( q.ansSelInfo[ currIndex ] ){
					return "<hr/>" + lang.ans.whyAns + "<br/>" + (q.ansSelInfo[ currIndex ][ stats.indexSelected[ currIndex] ] || q.ansInfo[ currIndex ]);
				}else{
					return "<hr/>" + lang.ans.whyAns + "<br/>" + (q.ansInfo[ currIndex ] );
				}
			},
			//setQResultDisplay(): IE Bug Fix. .q-result pops out when hidden. Thus, show() -> append() and hide() -> remove().
			setQResultDisplay = function( doAppend ){
				if( !isBadMSIE ){ return; }
				if( doAppend ){
					if( !$( ".q-result", currQuiz ).size() ){
						$(".q-prob", currQuiz ).append( $("<div/>").addClass( "q-result q-probArea" ).show() );
					}
				}
				else{
					$( ".q-result", currQuiz ).remove();
				}
			},
			displayFeedback = function( str ){
				setQResultDisplay(1);
				$( ".q-result", currQuiz ).html( str );
				if( !quit ){
					animateThis( $( ".q-result", currQuiz ), 1 );
				}
				else{
					$( ".q-result", currQuiz ).show();				
				}
			},
			// displayAnsResult(): Display's the answer result and information during the quiz.
			displayAnsResult = function( isFlashCard ){
				var currAns = getAns(), isUserAnsCorr = haveCorrAnsWithinRetries( isFlashCard ),
					show = "", toUni = settings.showHTML;
					
				currAns = ( isArray( currAns ) ) ? currAns.concat(): [ currAns ];					
				show = ( isUserAnsCorr ) ? lang.ans.praise :
								( lang.ans.corrAns + '<br/>' + getStrArrOfSeeableHTML( currAns, toUni ).join( '<br/>' ) );
			
				if( !isUserAnsCorr || quit ){ 
					if( settings.showWrongAns || quit ){
						show = getUserAnsForDisplay( toUni ) + "<hr/>" + show;						
					}
					if( settings.showAnsInfo ){
						show += getAnsInfoForDisplay();
					}
				}else{
					if( settings.alwaysShowAnsInfo ){
						show += getAnsInfoForDisplay();
					}
				}
				if( settings.showFeedback || quit ){ 
					displayFeedback( show );
				}
			},
			hasAnsSel = function( iProp, i ){
				return ( !!wordList[ q.prop[ iProp ] ][ i ].ansSel );
			},	
			rAnsSel = function( iProp, i ){
				var wAS = wordList[ q.prop[ iProp ] ][ i ].ansSel,
					j = rNum( ( isArray( wAS ) ) ? wAS.length : 0 );
				
				return { 
					index: j,
					value: ( isArray( wAS ) ) ? wAS[ j ] : wAS
				};
			},
			releaseButtonBlock = function(){
				if( settings.showFeedback ){
					$( ".q-check-btn", currQuiz ).attr( "disabled", !$( ".q-next-btn", currQuiz ).attr("disabled") );
				}
				else{
					$( ".q-next-btn", currQuiz ).attr( "disabled", false );
				}
				
			},
			setAnsSelInfoSelect = function( quizType ){
				if( !q.ansSelInfo[ currIndex ] ){
					return;
				}
				groupClass = { 
					"multi":".q-select:visible > option",
					"multiList":".q-ol-li:visible",
					"tf":".userInputArea > input"
				}[quizType];
				selectedClass = { 
					"multi":".q-select > option:selected",
					"multiList":"."+settings.activeClass,
					"tf":".userInputArea > input:checked"
				}[quizType];
				stats.indexSelected[ currIndex ] = $( groupClass, currQuiz ).index( $(selectedClass, currQuiz ) );
			},
			// buildQuizType{}: contains all the quiz types.
			//Quiz creation: The q object is filled with the questions, answers and answer selections(what the user can choose from). 
			//The user's input gets assigned to stats.problem[ currIndex ].userAnswer ( global inside quizMe ). 
			//Then nextMove() is called. Which checks to see if the user's input is contained in the answer, or is the array.
			buildQuizType = {
				// fillInTheBlank Quiz:	ques = ques, ansSel = Input text[ ans ].
				'fillInTheBlank' : function( iProp, flashVer ){
					var d;
					if( flashVer ){
						d = $( "<span/>" ).addClass( "clickThis userInputArea" ) //Enabled the checkBtn.
							.one( "click", function(){
								releaseButtonBlock();
							})
							.bind( "getUserAns", function(){ return 1; });								
					}
					else{
						d = $( '<input type="text"/>' ).addClass( "q-quesInput focusThis userInputArea" )
							.one( "click keypress", function(){
								releaseButtonBlock();
							})
							.bind( "getUserAns", function(){
								setUserAnswer( $.trim( $( ".q-quesInput", currQuiz ).val() ) );
							});
					}
					var i = q.indexMax[ iProp ], qMin = q.indexMin[ iProp ], wList = wordList[ q.prop[ iProp ] ], numOfQues = wList.length;
					
					while( numOfQues-- ){
						q.retryCount[ i ] = wList[ numOfQues ].retry || 0;
						q.ansInfo[ i ] = wList[ numOfQues ].ansInfo || ""; 
						q.ans[ i ] = wList[ numOfQues ].ans;
						q.ansSel[ i ] = qMin;
						q.ques[ i ] = ( wList[ numOfQues ].ques + settings.addToEnd );
						i--;
					}
					q.ansSel[ qMin ] = d;	//speed!! All the q.ansSel area reference to qMin. 
				},
				'flashCard' : function( iProp ){
					// flashCard Quiz:  ques = ques, ansSel =null .
					this.fillInTheBlank( iProp, true);
				},			
				'trueOrFalse' : function( iProp ){
					// trueOrFalse Quiz:
					// ques  = (typeof ans != bool) ? (real or fake ans)  : ques;
					// ansSel =  T / F ( radio );
					// If the answer is a bool, then there is no creation of a question: combining a right or wrong answers, to make a true or false statement.
					
					var d = $( ['<div><input type="radio" value="1" class="true-radio trueRadio"/><label class="trueRadio">', lang.quiz.tfTrue,
								'</label> <input type="radio" value="0" class="false-radio falseRadio"/><label class="falseRadio">', 
								lang.quiz.tfFalse, '</label></div>'].join('') );
					
					$( d ).addClass( "userInputArea" ).bind( "getUserAns", function(){
						var isTrue = $( ".true-radio", this ).attr( "checked" );
						
						setUserAnswer( lang.quiz[ ( isTrue ) ? "tfTrue" : "tfFalse" ] );
						setAnsSelInfoSelect( "tf" );
					});
					$( d ).children().one( "click", function(){
						releaseButtonBlock();
					});
					$(".trueRadio", d).click( function(){
						$( ".true-radio", currQuiz ).attr( "checked", true ); 
						$( ".false-radio", currQuiz ).attr( "checked", false ); 
					});
					$(".falseRadio", d).click( function(){
						$( ".true-radio", currQuiz ).attr( "checked", false ); 
						$( ".false-radio", currQuiz ).attr( "checked", true ); 
					});			
					var currAns, shouldAnsBeCorr = true, i = q.indexMax[ iProp ], 
						qMin = q.indexMin[ iProp ], wList = wordList[ q.prop[ iProp ] ], numOfQues = wList.length,
						toUni = settings.showHTML, tfEqual = lang.quiz.tfEqual, tfEnd = lang.quiz.tfEnd, rAS, cAS, result, isAnsReallyCorrect;

					while( numOfQues-- ){
						rAS = undefined;
						cAS = undefined;
						currAns = wList[ numOfQues ].ans;
						q.ansInfo[ i ] = wList[ numOfQues ].ansInfo || ""; 
						q.ansSel[ i ] = qMin;
						q.retryCount[ i ] = wList[ numOfQues ].retry || 0;
						q.ques[ i ] = wList[ numOfQues ].ques;
						if( typeof currAns != "boolean" ){
							shouldAnsBeCorr = rBool();
							if( shouldAnsBeCorr ){
								result = currAns;
							}else{
								if( hasAnsSel( iProp, numOfQues ) ){
									rAS = rAnsSel( iProp, numOfQues );
									result = rAS.value || currAns;
								}
								else{
									result = rAns(iProp);
								}
							}				
							isAnsReallyCorrect = ( !isArray( currAns ) || !isArray( result ) ) ? result == currAns : areArraysSame( result, currAns );
							q.ans[ i ] = lang.quiz[ ( isAnsReallyCorrect ) ? "tfTrue" : "tfFalse" ];
							q.ques[ i ] += tfEqual;
							q.ques[ i ] += getStrWithSeeableHTML( ( isArray( result ) ) ? result.join( ',' ) : result, toUni );
							q.ques[ i ] += tfEnd;
						}
						else{
							q.ans[ i ] = lang.quiz[ ( currAns ) ? "tfTrue" : "tfFalse" ];
						}
						if( wList[ numOfQues ].ansSelInfo ){
							cAS = wList[ numOfQues ].ansSelInfo.concat();
							if( cAS && rAS ){
								cAS = ( isArray( cAS ) ) ? cAS : [ cAS ];
								var qAI = q.ansInfo[ i ];
								q.ansSelInfo[ i ] = ( shouldAnsBeCorr ) ? [ qAI, cAS[ rAS.index ] || qAI ] : [ cAS[ rAS.index ] || qAI, q.ansInfo[ i ] ];
							}
						}
						i--;
					}
					q.ansSel[ qMin ] = d;
				},
				'multipleChoice' : function( iProp, olVer ){
					// multipleChoice Quiz: 
					// ques = ques;
					// ansSel = <select>[ <option> is the answer + <option> * settings.multiLen]
						var d;
						if( olVer ){
							d = $( '<div><ol class="q-ol"></ol></div>' );
							$( d ).one( "click", function(){
								releaseButtonBlock();
							});
						}
						else{
							d = $( '<div><select class="q-select"></select></div>' );
							$( ".q-select", d ).one( "click keypress", function( e ){
								releaseButtonBlock();
							});
						}
						$( d ).addClass( "userInputArea" ).bind( "getUserAns", function(){		
							if( olVer ){ 
								var elHtml = $("." + settings.activeClass, currQuiz).find(".q-ol-li-ansSel")[ settings.showHTML ? "text": "html" ]();
								setUserAnswer( $.trim(elHtml) );
								setAnsSelInfoSelect( "multiList" );
							}
							else{
								setUserAnswer( $( "option:selected", this ).text() );
								setAnsSelInfoSelect( "multi" );
							}
						});
						
						var i = q.indexMax[ iProp ], wList = wordList[ q.prop[ iProp ] ], ansPos, optHtml, val, len,
							j = wList.length, wListRange = getNumRange( j ), numOfQues = wList.length,
//Checks to see, if the request multiLen length can be supported.
							setLen = ( settings.multiLen > j ) ? j : settings.multiLen,
							toUni = settings.showHTML, hasAnsPosition = false, ansSelChoice = "", range, cASI, cProb, ansIndex, x; 
							
						var addClassFunc = function( e ){ 
								$( e.target ).closest( "li.q-ol-li" ).addClass( settings.hoverClass );
						}, 
						removeClassFunc = function( e ){
							$( e.target ).closest( "li.q-ol-li" ).removeClass( settings.hoverClass );
						},
						clickFunc = function( e ){					
							var qOL = $(e.target).closest("li.q-ol-li");
							qOL.addClass(settings.activeClass).siblings().removeClass(settings.activeClass).find(".q-radioBtn:checked").attr("checked", false);
							qOL.find(".q-radioBtn").attr("checked", true);
						};
							
					while (numOfQues--) {
						cProb = wList[numOfQues];
						optHtml = "";
						hasAnsPosition = false;
						cASI = cProb.ansSelInfo;
						cASI = ( cASI ) ? ( ( isArray( cASI ) ) ? cASI.concat() : [ cASI ] ) : undefined;
						if (hasAnsSel(iProp, numOfQues)) {
							val = [];
							if ( isArray( cProb.ansSel ) ) {
								var cAS = cProb.ansSel,
									k = cAS.length;
									
								while ( k-- ) {
									if (cAS[ k ] === null) {
										val[ k ] = cProb.ans;
										if( cASI ){
											cASI.splice( k, 1, cProb.ansInfo );
										}
										hasAnsPosition = true;
									} else {
										val[ k ] = cAS[k];
									}
								}
							} else {
								val[0] = cProb.ansSel;
							}							
							if (!hasAnsPosition) {
								val[val.length] = cProb.ans;
								if( cASI ){
									cASI[ cASI.length ] = cProb.ansInfo;
								}
							}
							len = val.length;
							range = getNumRange( len );
							if( !hasAnsPosition ){
								makeArrayRandom( range );
								if( cASI ){
									cASI = getArrayOfSwappedArrayElements( cASI, range );
								}
							}
							while (len--) {
								ansSelChoice = getStrWithSeeableHTML( val[ range[ len ] ], toUni );
								if( cASI ){
									cASI[ len ] = getStrWithSeeableHTML( cASI[ len ], toUni );
								}
								x = (olVer) ? ["<li class = 'q-ol-li'><input type='radio' class='q-radioBtn'/><span class='q-ol-li-ansSel'>", ansSelChoice, "</span></li> "].join("") : ["<option>", ansSelChoice, "</option> "].join("");
								optHtml = x + optHtml;
							}
						} else {
							var randAns = wListRange.concat(); 
							// Get rids of the answer Index from randAns. 
							randAns.splice(numOfQues, 1);
							
							len = setLen;
							ansPos = rNum(len);
							while (len--) {
								var randIndex = rNum(randAns.length),
								ansSelIndex = (len == ansPos) ? numOfQues: (randAns.splice(randIndex, 1) || 0);
								ansSelChoice = getStrWithSeeableHTML(wList[ansSelIndex].ans, toUni);
								x = (olVer) ? ["<li class = 'q-ol-li'><input type='radio' class='q-radioBtn'/><span class='q-ol-li-ansSel'>", ansSelChoice, "</span></li> "].join("") : ["<option>", ansSelChoice, "</option> "].join("");
								optHtml = x + optHtml;
							}
						}
						// below: innerHTML is 2x faster than $.html() but IE generates corrrupt option html
						if (isBadMSIE && !olVer) {
							$("select", d).html(optHtml);
						} else {
							$((olVer) ? ".q-ol": "select", d)[0].innerHTML = optHtml;
						}
						if (isBadMSIE && olVer) { 
						//Fixes IE CSS BUG. No spaces with list-style-position: inline.
							$(".q-ol-li", d).prepend("&nbsp;");
						}
						if( olVer ){
							$( "li.q-ol-li", d ).bind( "click", clickFunc )
								.bind( "mouseover", addClassFunc )
								.bind( "mouseout", removeClassFunc );
						}
						q.retryCount[i] = cProb.retry || 0;
						q.ans[i] = cProb.ans;
						q.ansInfo[i] = cProb.ansInfo || "";
						q.ansSel[i] = d.clone(true);
						q.ansSelInfo[ i ] = ( cASI ) ? cASI.concat() : cASI;
						q.ques[i] = cProb.ques;
						i--;
					} 
				},
				'multipleChoiceOl' : function( iProp ){
					// multipleChoiceOlQuiz: ques = ques, ansSel = ol [ <list> real ans + <list> * settings.multipleChooseLength] 
					this.multipleChoice( iProp, true);
				}
			},
			// makeQuizRandom(): If randomizeQ is false, it will randomize the each quiz type. 
			// But if randomizeQ is true, it will randomize the q object.
			makeQuizRandom = function( randomizeQ ){
				if( randomizeQ ){
					var numArr = getNumRange( totalQuesLen ), i = numArr.length,
						temp = { ans: [], ansSel: [], ansInfo: [], ques: [] }, nArrI, qAS;
					
					makeArrayRandom( numArr );
					i = numArr.length;

					while( i-- ){
						nArrI = numArr[ i ];
						qAS = q.ansSel[ nArrI ];
						
						temp.ans[i] = q.ans[ nArrI ];
						temp.ansInfo[i] = q.ansInfo[ nArrI ] || "";
						temp.ansSel[i] = ( typeof qAS !== "number" ) ? qAS : q.ansSel[ qAS ];
						temp.ques[i] = q.ques[ nArrI ];
					}
					q.ans = temp.ans.concat();
					q.ansInfo = temp.ansInfo.concat();
					q.ansSel = temp.ansSel.concat();					
					q.ques = temp.ques.concat();
				}
				else{				
					var iProp = q.prop.length;
					while( iProp-- ){
						makeArrayRandom( wordList[ q.prop[ iProp ] ] ); 
					}					
				}
			},
			makeQuizType = function(){ 
			//iProp is a pointer to the quizType min and max index.
				var iProp = q.prop.length, currProp = "", p, allP = settings.allQuizType,
					shortProp = {
						fill:"fillInTheBlank", cards:"flashCard", tf:"trueOrFalse", multi:"multipleChoice", multiList:"multipleChoiceOl"
					};
					
				if( settings.random ){ makeQuizRandom();}
				while( iProp-- ){
					p = allP || q.prop[ iProp ];
					currProp = shortProp[ p ] || p;
					buildQuizType[ currProp ]( iProp );
				}
				if( settings.allRandom ){ 
					makeQuizRandom(true);
				}
			},
			// Creates and shows the new quiz.
			createNewQuizProb = function(){
				stats.reset();
				makeQuizType();
				changeQuizInfo();
				changeProb();
				animateThis( $( currQuiz ), 1, sendStatus );
			},
			quitQuiz = function(){
				currIndex = 0; //Used to make updateReviewIndex start at 0.
				quit = true; 
				disableMenuBar();
				gameOver();				
			},
			deleteQuiz = function(){
				animateThis( $( currQuiz ), 0, function(){
					$( currQuiz ).remove();
				});
				q = {};
			},
			setToBeginningView = function(){
				$( ".q-gameOver, .q-result, .q-review-menu", currQuiz).hide();
				$( ".q-ans, .q-help, .q-help-menu, .q-prob", currQuiz ).show();
				$( ".q-check-btn", currQuiz )[ ( settings.showFeedback)?"show":"hide"]().attr( "disabled", true );
				$( ".q-next-btn", currQuiz ).attr( "disabled", settings.showFeedback );
				$( ".q-quit-btn, .q-help-btn", currQuiz ).attr( "disabled", false );
			},
			reStartQuiz = function(){
				//IE likes to hide on restart.
				if( !isBadMSIE || ( isBadMSIE && settings.fxType == 3 ) ){
					animateThis( $( currQuiz ), 0 );
				}
				setToBeginningView();
				quit = false;
				currIndex = 0;
				createNewQuizProb();
			},
			setAnimation = function( s ){
				var spd = s.fxSpeed, spdType = { slow: 600, fast: 200, normal: 400 };
				
				s.fxSpeed = isNaN( spd ) ? ( spdType[ spd ] || 400 ) : spd;
				if( s.fxCode ){
					var custAnimatePos = animateType.length;
					animateType[ custAnimatePos ] = s.fxCode;
					s.fxType = custAnimatePos;
				}
			},
			checkAnsSelInfo = function( ansInfo, ansSel, ansSelInfo ){
				var a = isArray( ansSel ), b = isArray( ansSelInfo );
				
				if( ansInfo === undefined ){
					return lang.err.ansInfoNotDefined;
				}
				if( ( a || b ) && (a != b || ansSel.length !== ansSelInfo.length)){
					return lang.err.badAnsSelInfoLen;
				}
			},
			// checkQTypeKeys(): Checks the keys for each question, wordList[i]. Returns error if unknown.
			checkQTypeKeys = function( quesKeys ){
				var badKey = "", i = quesKeys.length, temp,
					validKeys = { ques:1, ans:1, ansInfo:1, ansSel:1, ansSelInfo:1, retry:1 };
				
				if( !i ){ badKey = lang.err.noQues; }
				while( i-- ){
					if( badKey ){ break;}
					temp = quesKeys[ i ];
					for( var currKey in temp ){
						if( temp.hasOwnProperty( currKey ) ){
							if( !validKeys[ currKey ] ){
								badKey = currKey + lang.err.badKey;
								break;
							}
							if( currKey == "ansSelInfo" ){
								badKey = checkAnsSelInfo( temp.ansInfo || undefined, temp.ansSel || undefined, temp.ansSelInfo );
								break;
							}
						}
					}
				}
				return badKey;
			},
			hasBadSyntax = function( wNames ){
				var badKey = "", goodKeys = [],
					validKeys = { 
						fillInTheBlank:1, flashCard:1, trueOrFalse:1, multipleChoice:1, multipleChoiceOl:1,
						fill:1, cards:1, tf:1, multi:1, multiList:1
					};
				
				for( var prop in wNames ){
					if( wNames.hasOwnProperty( prop ) ){
						if( !validKeys[ prop ] ){
							badKey = lang.err[ ( typeof wNames === "object" ) ? "badQType" : "lang.err.notObj" ];
						}
						if( !badKey ){
							badKey = ( isArray( wNames[prop] ) ) ? checkQTypeKeys( wNames[prop] ) : lang.err.notArr;
						}
						if( badKey ){ 
							badKey = prop + " -> " + badKey;
							break; 
						}
					}
				}
				
				if( !prop ){ badKey = lang.err.noQType; }
				if( !badKey && settings.allQuizType && !validKeys[ settings.allQuizType ] ){ 
					badKey = settings.allQuizType + " -> " + lang.err.badQType; 
				}
				return badKey;
			},
			checkForErrors = function( obj ){
				var err = ( obj ) ? hasBadSyntax( obj ) : lang.err.noQues;
				if( err ){ 
					err = "jQuizMe " + lang.err.error + ": " + err;
					$( stickyEl ).text( err );
					throw new Error( err );
				}
			},
			convertArrToObj = function( obj ){
				var tempObj = {};
				tempObj[ settings.allQuizType || settings.quizType ] = obj;
				return tempObj;
			},
			// start(): This is the first function called.
			start = function(){
				wordList = ( isArray( wordList ) ) ? convertArrToObj( wordList ) : wordList;
				if( settings.performErrorChecking ){
					checkForErrors( wordList );
				}		
				setAnimation( settings );
				q.prop = getObjProps( wordList );
				createQuizEl();
				createNewQuizProb();
				return currQuiz;
			}();	// Auto starts.
		});
	};
	$.fn.jQuizMe.version = "2.2";
})(jQuery);$( ".q-counter", currQuiz).text( 'Question' (currIndex + 1) + ' of ' + totalQuesLen );