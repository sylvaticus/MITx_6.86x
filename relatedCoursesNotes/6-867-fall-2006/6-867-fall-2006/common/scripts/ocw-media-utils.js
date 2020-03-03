/* OCW Media Utility JavaScript file
    Includes 
	         Custom functions related to Media(Audio/Video)
	         SWFObject v2.2
 */ 
 
 /**
 * SWFObject v1.5: Flash Player detection and embed - http://blog.deconcept.com/swfobject/
 *
 * SWFObject is (c) 2007 Geoff Stearns and is released under the MIT License:
 * http://www.opensource.org/licenses/mit-license.php
 *
 */
if(typeof deconcept=="undefined"){var deconcept=new Object();}if(typeof deconcept.util=="undefined"){deconcept.util=new Object();}if(typeof deconcept.SWFObjectUtil=="undefined"){deconcept.SWFObjectUtil=new Object();}deconcept.SWFObject=function(_1,id,w,h,_5,c,_7,_8,_9,_a){if(!document.getElementById){return;}this.DETECT_KEY=_a?_a:"detectflash";this.skipDetect=deconcept.util.getRequestParameter(this.DETECT_KEY);this.params=new Object();this.variables=new Object();this.attributes=new Array();if(_1){this.setAttribute("swf",_1);}if(id){this.setAttribute("id",id);}if(w){this.setAttribute("width",w);}if(h){this.setAttribute("height",h);}if(_5){this.setAttribute("version",new deconcept.PlayerVersion(_5.toString().split(".")));}this.installedVer=deconcept.SWFObjectUtil.getPlayerVersion();if(!window.opera&&document.all&&this.installedVer.major>7){deconcept.SWFObject.doPrepUnload=true;}if(c){this.addParam("bgcolor",c);}var q=_7?_7:"high";this.addParam("quality",q);this.setAttribute("useExpressInstall",false);this.setAttribute("doExpressInstall",false);var _c=(_8)?_8:window.location;this.setAttribute("xiRedirectUrl",_c);this.setAttribute("redirectUrl","");if(_9){this.setAttribute("redirectUrl",_9);}};deconcept.SWFObject.prototype={useExpressInstall:function(_d){this.xiSWFPath=!_d?"expressinstall.swf":_d;this.setAttribute("useExpressInstall",true);},setAttribute:function(_e,_f){this.attributes[_e]=_f;},getAttribute:function(_10){return this.attributes[_10];},addParam:function(_11,_12){this.params[_11]=_12;},getParams:function(){return this.params;},addVariable:function(_13,_14){this.variables[_13]=_14;},getVariable:function(_15){return this.variables[_15];},getVariables:function(){return this.variables;},getVariablePairs:function(){var _16=new Array();var key;var _18=this.getVariables();for(key in _18){_16[_16.length]=key+"="+_18[key];}return _16;},getSWFHTML:function(){var _19="";if(navigator.plugins&&navigator.mimeTypes&&navigator.mimeTypes.length){if(this.getAttribute("doExpressInstall")){this.addVariable("MMplayerType","PlugIn");this.setAttribute("swf",this.xiSWFPath);}_19="<embed type=\"application/x-shockwave-flash\" src=\""+this.getAttribute("swf")+"\" width=\""+this.getAttribute("width")+"\" height=\""+this.getAttribute("height")+"\" style=\""+this.getAttribute("style")+"\"";_19+=" id=\""+this.getAttribute("id")+"\" name=\""+this.getAttribute("id")+"\" ";var _1a=this.getParams();for(var key in _1a){_19+=[key]+"=\""+_1a[key]+"\" ";}var _1c=this.getVariablePairs().join("&");if(_1c.length>0){_19+="flashvars=\""+_1c+"\"";}_19+="/>";}else{if(this.getAttribute("doExpressInstall")){this.addVariable("MMplayerType","ActiveX");this.setAttribute("swf",this.xiSWFPath);}_19="<object id=\""+this.getAttribute("id")+"\" classid=\"clsid:D27CDB6E-AE6D-11cf-96B8-444553540000\" width=\""+this.getAttribute("width")+"\" height=\""+this.getAttribute("height")+"\" style=\""+this.getAttribute("style")+"\">";_19+="<param name=\"movie\" value=\""+this.getAttribute("swf")+"\" />";var _1d=this.getParams();for(var key in _1d){_19+="<param name=\""+key+"\" value=\""+_1d[key]+"\" />";}var _1f=this.getVariablePairs().join("&");if(_1f.length>0){_19+="<param name=\"flashvars\" value=\""+_1f+"\" />";}_19+="</object>";}return _19;},write:function(_20){if(this.getAttribute("useExpressInstall")){var _21=new deconcept.PlayerVersion([6,0,65]);if(this.installedVer.versionIsValid(_21)&&!this.installedVer.versionIsValid(this.getAttribute("version"))){this.setAttribute("doExpressInstall",true);this.addVariable("MMredirectURL",escape(this.getAttribute("xiRedirectUrl")));document.title=document.title.slice(0,47)+" - Flash Player Installation";this.addVariable("MMdoctitle",document.title);}}if(this.skipDetect||this.getAttribute("doExpressInstall")||this.installedVer.versionIsValid(this.getAttribute("version"))){var n=(typeof _20=="string")?document.getElementById(_20):_20;n.innerHTML=this.getSWFHTML();return true;}else{if(this.getAttribute("redirectUrl")!=""){document.location.replace(this.getAttribute("redirectUrl"));}}return false;}};deconcept.SWFObjectUtil.getPlayerVersion=function(){var _23=new deconcept.PlayerVersion([0,0,0]);if(navigator.plugins&&navigator.mimeTypes.length){var x=navigator.plugins["Shockwave Flash"];if(x&&x.description){_23=new deconcept.PlayerVersion(x.description.replace(/([a-zA-Z]|\s)+/,"").replace(/(\s+r|\s+b[0-9]+)/,".").split("."));}}else{if(navigator.userAgent&&navigator.userAgent.indexOf("Windows CE")>=0){var axo=1;var _26=3;while(axo){try{_26++;axo=new ActiveXObject("ShockwaveFlash.ShockwaveFlash."+_26);_23=new deconcept.PlayerVersion([_26,0,0]);}catch(e){axo=null;}}}else{try{var axo=new ActiveXObject("ShockwaveFlash.ShockwaveFlash.7");}catch(e){try{var axo=new ActiveXObject("ShockwaveFlash.ShockwaveFlash.6");_23=new deconcept.PlayerVersion([6,0,21]);axo.AllowScriptAccess="always";}catch(e){if(_23.major==6){return _23;}}try{axo=new ActiveXObject("ShockwaveFlash.ShockwaveFlash");}catch(e){}}if(axo!=null){_23=new deconcept.PlayerVersion(axo.GetVariable("$version").split(" ")[1].split(","));}}}return _23;};deconcept.PlayerVersion=function(_29){this.major=_29[0]!=null?parseInt(_29[0]):0;this.minor=_29[1]!=null?parseInt(_29[1]):0;this.rev=_29[2]!=null?parseInt(_29[2]):0;};deconcept.PlayerVersion.prototype.versionIsValid=function(fv){if(this.major<fv.major){return false;}if(this.major>fv.major){return true;}if(this.minor<fv.minor){return false;}if(this.minor>fv.minor){return true;}if(this.rev<fv.rev){return false;}return true;};deconcept.util={getRequestParameter:function(_2b){var q=document.location.search||document.location.hash;if(_2b==null){return q;}if(q){var _2d=q.substring(1).split("&");for(var i=0;i<_2d.length;i++){if(_2d[i].substring(0,_2d[i].indexOf("="))==_2b){return _2d[i].substring((_2d[i].indexOf("=")+1));}}}return "";}};deconcept.SWFObjectUtil.cleanupSWFs=function(){var _2f=document.getElementsByTagName("OBJECT");for(var i=_2f.length-1;i>=0;i--){_2f[i].style.display="none";for(var x in _2f[i]){if(typeof _2f[i][x]=="function"){_2f[i][x]=function(){};}}}};if(deconcept.SWFObject.doPrepUnload){if(!deconcept.unloadSet){deconcept.SWFObjectUtil.prepUnload=function(){__flash_unloadHandler=function(){};__flash_savedUnloadHandler=function(){};window.attachEvent("onunload",deconcept.SWFObjectUtil.cleanupSWFs);};window.attachEvent("onbeforeunload",deconcept.SWFObjectUtil.prepUnload);deconcept.unloadSet=true;}}if(!document.getElementById&&document.all){document.getElementById=function(id){return document.all[id];};}var getQueryParamValue=deconcept.util.getRequestParameter;var FlashObject=deconcept.SWFObject;var SWFObject=deconcept.SWFObject;
 
// For IPad Scroll  
   function isMobile(){
       var index = navigator.appVersion.toLowerCase().indexOf("mobile");       
       return (index > -1);
    }    
     
    function IpadScroller(){       
        if(isMobile()){ 
            $('.scroll-pane').each(function(index){
                 if ($(this).attr('display')!='hidden')        $(this).jScrollPane();});

            $("a.scroll_ipad",$("#tabs ul li")).click(function() {	    
	    	var activeTab = $(this).attr("href"); //Find the href attribute value to identify the active tab + content
	    	scrollRefresh(activeTab);	    
		    });           
        }        
    }
    
    function scrollRefresh(anchorTag){$('#'+anchorTag.substring(1)).jScrollPane();}    
    
/* POPUP Code When you click on a link with class of poplight and the href starts with a # */
var player;
var cues;
var annots;
var count;
$(document).ready(function() {	
$('a.poplight').click(function() {
    
    var popID = $(this).attr('rel'); //Get Popup Name
    var popURL = $(this).attr('href'); //Get Popup href to define size

    //Pull Query & Variables from href URL
    var query= popURL.split('?');
    var dim= query[1].split('&');
    var popWidth = dim[0].split('=')[1]; //Gets the first query string value

	//Define margin for center alignment (vertical   horizontal) - we add 80px to the height/width to accomodate for the padding  and border width defined in the css
    //var popMargTop = ($('#' + popID).height() + 325 + 80) / 2;
    var popMargTop = (388 + 80) / 2;
    var popMargLeft = ($('#' + popID).width()+ 80) / 2;
	if (cues != undefined){	    
		popWidth = 994;		
		popMargLeft = 530;
	}	
	var interactiveTranscriptContainer =$(this).attr('onclick').split(',')[1].split("'")[1];	
	interactiveTranscriptDivName = interactiveTranscriptContainer + "transcript1";	
	
	if($('#' + interactiveTranscriptDivName).length > 0){		
		popMargTop = (388+140+80) / 2; // pop up height, PlugIn height Padding and border width
		plugin_params={};
		p3_external_stylesheet = window.location.host + "/styles/three_play.css";   		
		plugin_params[interactiveTranscriptContainer] = threePlayParams[interactiveTranscriptContainer];
		plugin_params[interactiveTranscriptContainer]['transcript']['height'] = 140;
		three_play_obj = $(document)[0].defaultView.P3
		if ( three_play_obj != undefined){three_play_obj.init(plugin_params,""); }		          
		}
		
    //Fade in the Popup and add close button
	close_btn = "/images/close.png";	
	if (window.location.hostname.indexOf("dspace") >= 0)
	{		
		var scripts = document.getElementsByTagName('link');	
		txt = scripts[0].getAttribute('href');
		
		if ((txt !=null) && (txt.indexOf("/common/") >= 0)){			
			prepend = txt.split('/common/');			
			close_btn = txt.split('/common/')[0] + '/common' + close_btn;						
			}		
	}	
   $('#' + popID).fadeIn().css({ 'width': Number( popWidth ) }).prepend('<a href="#" class="close"><img src="'+close_btn +'" class="video_popup_btn_close" title="Close Window" alt="Close" /></a>');

  

    //Apply Margin to Popup
    $('#' + popID).css({
        'margin-top' : -popMargTop,
        'margin-left' : -popMargLeft
    });

    //Fade in Background
    $('body').append('<div id="video_popup_fade"></div>'); //Add the fade layer to bottom of the body tag.
    $('#video_popup_fade').css({'filter' : 'alpha(opacity=80)'}).fadeIn(); //Fade in the fade layer - .css({'filter' : 'alpha(opacity=80)'}) is used to fix the IE Bug on fading transparencies       
    return false;
});

//Close Popups and Fade Layer
$('a.close, #video_popup_fade').live('click', function() { //When clicking on the close or fade layer...
    $('#video_popup_fade , .popup_block').fadeOut(function() {
        $('#video_popup_fade, a.close').remove();  //fade them both out
    });
    jwplayer(player).stop();
    return false;
});
});

/* OCW Video Player code */

var pos;
var stop_marker = 1000000; 
var sub_titles = [];

var subtitleElement = document.getElementById('subtitles');

function toSeconds(t) {
    var s = 0.0
    if(t) {
        var p = t.split(':');
        for(i=0;i<p.length;i++)
            s = s * 60 + parseFloat(p[i].replace(',', '.'))
    }
    return s;
}

function strip(s) {
    if ( s==null )
        return s;
    return s.replace(/^\s+|\s+$/g,"");
}

// This function loads the subtitles(caption) when the "load captions outside the player" option is selected in cms and this functionality is not applicable in offline version.
function loadSubtitles(subtitleElement) {
    var videoId = subtitleElement.attr('data-video');
    var srt = subtitleElement.attr('alt');
    subtitleElement.text('');
    srt = srt.replace(/\r\n|\r|\n/g, '\n')
    srt = strip(srt);    
    sub_titles = [];
    
    var srt_ = srt.split('\n\n');
    for(s in srt_) {
        st = srt_[s].split('\n');
        if(st.length >=2) {
          n = st[0];
          i = strip(st[1].split(' --> ')[0]);
          o = strip(st[1].split(' --> ')[1]);
          t = st[2];
		  if(t==undefined){
		     t ="";
		    }
          if(st.length > 2) {
            for(j=3; j<st.length;j++)
              t += '\n'+st[j];			  
            }
          is = toSeconds(i);
          os = toSeconds(o);
          sub_title = {'start':is, 'end': os, 'text': t}
          sub_titles[parseInt(n)] = sub_title;
        }
     }    
    return;
}

var cc_on = 1;

$(document).ready(function() {
    // Load Subtitles
    $('.srt').each(function() {
         var subtitleElement = $(this);
         loadSubtitles(subtitleElement);
         return;
      });
	$("#cc-control").click(function(){
		if(cc_on == 0){
			$("#subtitles").animate( { height:"49px",opacity: 5 }, { queue:false, duration:500 } );
			cc_on = 1;
			document.getElementById("captions").style.backgroundImage = "url(/images/cc_on.PNG)";			
		    } 
		else {
			$("#subtitles").animate( { height:"0px", opacity: 0 }, { queue:false, duration:500 } );
			cc_on = 0;			
			document.getElementById("captions").style.backgroundImage = "url(/images/cc_off.PNG)";			
		   }
		
	   });
});    

function get_subtitle(position){
	for(i = 1; i <= sub_titles.length; i ++){
		if((sub_titles[i]['start'] <= position) && (sub_titles[i]['end'] >= position)){
			subtitle = sub_titles[i]['text'];
			return subtitle;
		}
	}	
	return '';
}

function timeMonitor(obj){
    time = obj.position;
    current_position = time;	
    if (current_position > stop_marker){	   
        jwplayer(player).stop();
       }
	 if (document.getElementById('subtitles') != null){document.getElementById('subtitles').innerHTML = get_subtitle(current_position).replace("\n", "<br>");}  
}

function setSlides(now,cues,container_id){
	for (var cue in cues) {
		if (!cues.hasOwnProperty (cue)) continue;
		cue = cues[cue];
		if (cue[0] == Math.round(now) ) {
		  //console.log ('cue', cue[0], cue[1], cue[2], cue[3]);
		  do_cue (cue,container_id);
		  // NO break; (want to cue both systems)
		}
	  }
}

// Only change slide if different from before, so user can change it.
var last = null;
var prev_id = null;
function do_cue (cue,container_id) {
  if (last != cue[2]) {  
    last = cue[2] ;
    prev_id = container_id;	
    change (cue[1], cue[2],container_id,false);
  }
}

function change (path, page, container_id, relative) {	
	if (relative){	
        if (prev_id != container_id){
			last = parseInt(document.getElementById ('numbers_'+container_id).innerHTML) - 1;
			prev_id = container_id;
		}
		page += last
		if (page >= count) page -= 1		
		if (page < 0) {page = 0; }
	    else { last = page; }				
		if (page <= count){
		      if (cues[page] != undefined){
					path = cues[page][1];                        
				   }
               else{return}	
			}			
	     }			
	document.getElementById ('images_'+container_id).src = path 
	document.getElementById ('annots_'+container_id).innerHTML = annots[page] || '';  
	document.getElementById ('numbers_'+container_id).innerHTML = page+1;	
	
}

function prev (container_id) {
  change ('', -1, container_id, true);
}
function next (container_id) {
  change ('', +1, container_id, true);
}

var oldWrapper = null;
var oldCode;

function reset_video_wrapper_markup(theOldWrapper, theOldCode) { 
	document.getElementById(theOldWrapper).innerHTML = theOldCode;
}

function initialize_video_wrapper_markup(theWrapper) {
    // Live version
    if (oldWrapper != null) { reset_video_wrapper_markup(oldWrapper, oldCode); }

	oldWrapper = theWrapper; 
	oldCode = document.getElementById(oldWrapper).innerHTML; 
}


// This functions provides jump back to given position functionality and it is not applicable in offline version.
function playerJumpTo(pos){jwplayer(player).seek(pos);}

// This functions provides Adding "jump back 5 seconds button" functionality and it is not applicable in offline version.
function back5Seconds(){jwplayer(player).seek(current_position - 6);}

function ocw_embed_media(container_id, media_url, provider, page_url, image_url, captions_file){
    //set jwplayer options for setup function
	var options = set_jwplayer_options(container_id, media_url, provider, page_url, image_url, 0, 0, captions_file);
 
    //set the autostart property false to not start the video automatically
    options['autostart'] = 'false';
	
   // setting up the JWPlayer and events 
    jw_player = jwplayer(container_id).setup(options); 
    set_jwplayer_events(jw_player,container_id,0); 
   
}

// This function is used to embed chaptered media on pages
function ocw_embed_chapter_media(container_id, media_url, provider, page_url, image_url, start, stop, captions_file){
    //set jwplayer options for setup function
	var options = set_jwplayer_options(container_id, media_url, provider, page_url, image_url, start, stop, captions_file);
 
    //set the autostart property false to not start the video automatically
    options['autostart'] = 'false';    	
	
    // setting up the JWPlayer and events and seek videos up to the starting point	
    jw_player = jwplayer(container_id).setup(options);
	set_jwplayer_events(jw_player,container_id,start); 
}

function load_media_chapter(media_url,provider, page_url, image_url, start, stop, captions_file){
    //set jwplayer options for setup function
    var options = set_jwplayer_options('embed1',media_url, provider, page_url, image_url, start, stop, captions_file);
 
    //set the autostart property true to start the video automatically
    options['autostart'] = 'true';
	
	// setting up the JWPlayer and events and seek videos up to the starting point
	jw_player = jwplayer('embed1').setup(options); 
    set_jwplayer_events(jw_player,'embed1',start); 
}

function load_multiple_media_chapter(theWrapper,container_id,thePlayer,media_url,provider, page_url, image_url, start, stop, captions_file){
    initialize_video_wrapper_markup(theWrapper);
  
    //set jwplayer options for setup function
    var options = set_jwplayer_options(container_id, media_url, provider, page_url, image_url, start, stop, captions_file);
    cues = undefined; 
    //set the autostart property true to start the video automatically
    options['autostart'] = 'true';
		
	// setting up the JWPlayer and events and seek videos up to the starting point
	jw_player = jwplayer(container_id).setup(options);
	set_jwplayer_events(jw_player,container_id,start); 
	}

function scholar_video_popup(theWrapper,container_id,thePlayer,media_url,provider, page_url, image_url, start, stop, captions_file){
    // initialize the video container inner html 
     initialize_video_wrapper_markup(theWrapper);
    
    //set jwplayer options for setup function
    var options = set_jwplayer_options(container_id, media_url, provider, page_url, image_url, start, stop, captions_file);	
	
	// setting up the JWPlayer and events and seek videos up to the starting point
     jw_player = jwplayer(container_id).setup(options);
	 set_jwplayer_events(jw_player,container_id,start);       
}

// This function renders HFH specific videos. This function does not exist in offline version as HFH courses are not downloaded.
function HFH_video_popup(theWrapper,container_id,thePlayer,media_url,provider, page_url, image_url, start, stop, captions_file,entryID){
    initialize_video_wrapper_markup(theWrapper);
	
	//set jwplayer options for setup function
    var options = set_jwplayer_options(container_id, media_url, provider, page_url, image_url, start, stop, captions_file);
	
    // setting up the JWPlayer and events and seek videos up to the starting point	 
    jw_player = jwplayer(container_id).setup(options);
	set_jwplayer_events(jw_player,container_id,start); 	
    }

function video_popup(theWrapper,container_id,thePlayer,media_url,provider, page_url, image_url, start, stop, captions_file){
    // initialize the video container inner html 
    initialize_video_wrapper_markup(theWrapper);
    
    //set jwplayer options for setup function
    var options = set_jwplayer_options(container_id,media_url, provider, page_url, image_url, start, stop, captions_file);	
		
	// setting up the JWPlayer and events and seek videos up to the starting point
    jw_player = jwplayer(container_id).setup(options);	
	cues = window["cues_" + container_id];
	count = window["slides_" + container_id + "_count"];
	annots =  window["annots_" + container_id];
	set_jwplayer_events(jw_player,container_id,start);   
}
function setMultipleCaptionsOld(captions){							//Patch to add multiple caption for existing videos, needs to be removed before Full Site Republish 
  // getting the array of caption file path and language
	var caption_obj = jQuery.parseJSON(captions);
	var captions_lang = Object.keys(caption_obj);
	var caption_URLs=new Array();

	for (var i = 0; i < captions_lang.length; i++) {
		lang = captions_lang[i];    
		caption_file = caption_obj[lang];
		track = {file: caption_file,kind: "captions",label: lang};
		caption_URLs.push(track);
	}
	return caption_URLs;
}

function setMultipleCaptions(captions){
  // getting the array of caption file path and language
	var captions_lang = Object.keys(captions);	
	var caption_URLs=new Array();

	for (var i = 0; i < captions_lang.length; i++) {
		lang = captions_lang[i];		
		caption_file = captions[lang];		
		track = {file: caption_file,kind: "captions",label: lang};
		caption_URLs.push(track);
	}
	return caption_URLs;
}
function set_jwplayer_events(jw_player,container_id,start){
     jw_player.onReady(function() {	current_position = 0;	                                		
                                    var firstPosition = start;
                                    var isFirstPositionSet = false;
	                                player = document.getElementById(container_id);	                                
                                    if(start != 0){jw_player.onPlay(function () { if (isFirstPositionSet == false) { jw_player.seek(firstPosition); isFirstPositionSet = true; } });}
                                    jw_player.onTime(function(event){
										timeMonitor(event);                                       								
				                        if (cues !=undefined){
										     setSlides(event.position,cues,container_id);}				    
				                        });										
								   });        	 
       }

function get_modified_id(container_id){
	var index = container_id.indexOf('_chapter'); // To change the scholar chapter embed container id
	if(index > -1 ) {container_id = container_id.substring(0, index ? index :container_id.length);}        
    return container_id;
}	   
	   

function get_caption_details(container_id) { 
	//getting caption file path and language details
   	container_id = get_modified_id(container_id)
	if(document.getElementById("caption_" + container_id)!= null){	//Patch to add multiple caption for existing videos, needs to be removed before Full Site Republish
		var	captions_info  = document.getElementById("caption_" + container_id).innerHTML;
		var caption_url = setMultipleCaptionsOld(captions_info.replace(/'/ig,'"'));		
		return caption_url;
	}	
	var captions_info = window["caption_" + container_id];          
	if(captions_info != undefined){  
		var caption_url = setMultipleCaptions(captions_info);
		return caption_url;	}
	return;
    }
	
function set_jwplayer_options(container_id, media_url, provider, page_url, image_url, start, stop, captions_file){
    
	var autostart = 'false';
    if (start !=0){autostart = 'true';}
    
	
	// Sharing code does not exist in offline version as it is not applicable there
    var sharing_embed_prefix = '<iframe src="';
    var sharing_embed_suffix = '" height=325 width=545 frameborder=0></iframe>';
    
    if (provider == 'rtmp'){
    	techTvId = image_url.split('/').reverse()[1];
        sharing_embed_prefix += 'https://techtv.mit.edu/embeds/' + techTvId +'?size=large&amp;custom_width=530&amp;custom_height=325&amp;'
		var sharing_embed_string = sharing_embed_prefix + sharing_embed_suffix
		media_url = 'rtmp://s1cgf0x4vkgx2w.cloudfront.net/cfx/st/' + media_url  //set the URL for TechTV videos 
    }
    else{var sharing_embed_string =  sharing_embed_prefix + media_url + sharing_embed_suffix; }     
    

    var sharing_params =    {   link: 'https://ocw.mit.edu' + page_url,
                                code: encodeURI(sharing_embed_string)
                            };    
	
	
    var track = {};
    if (captions_file != null && captions_file !=''){
		track = [{file: captions_file,kind: "captions",label: "On"}]; //for case of existing inline_embeds 
		var caption_info = get_caption_details(container_id); // for all mediaresource_embeds
		if (caption_info != {} && caption_info != undefined){
			track =caption_info;
			}
       }
    
    if (stop==0){stop=1000000;}
    stop_marker = stop;  
    
    return  {allowscriptaccess: 'always',
            primary: 'html5',
			height: 325,
            width: 530,            
            file: media_url,
			sharing: sharing_params,
            autostart: autostart, 
            tracks: track,	
			startparam: "start",
            image: image_url,          
			ga: {},
            captions: {back: false, color: 'FFFFFF', fontsize: 15}
    };
}