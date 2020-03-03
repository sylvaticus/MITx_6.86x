
var depth = window.location.pathname.split('/').indexOf('index.htm') - window.location.pathname.split('/').indexOf('contents');
var rootPath = '';
for(var i=1; i<=depth;i++){
rootPath += '../'; 
}
var path = rootPath + 'common/images/jquerybubblepopup-theme'; 

// Common function to set the BubblePopup properties
	function setBubblePopupProprties(tag,innerHtml){
	$(tag).CreateBubblePopup({
										position : 'top',
										tail	 : {align: 'middle'},
                                        alwaysVisible: false,
                                        selectable: true,
                                        innerHtml :innerHtml,
                                        innerHtmlStyle: {
                                                            color:'#000000', 
                                                            'text-align':'left'
                                                        },
                                        themeName: 	'grey',
                                        themePath: 	path,
                                        themeMargins: {
                                                        total: '13px',
                                                        difference: '2px'
													}         
							});  
	}

$(document).ready(function()
{
	var bubblePropertiesMap={ 
// Match for amazon logo
//'a_logo_17.gif': 'When you purchase this book (or other media) from Amazon.com,<b> MIT OpenCourseWare will receive up to 10% of this purchase</b> and any other purchases you make during that visit. <a href="https://ocw.mit.edu/donate/support-ocw-by-shopping-at-amazon/">Learn more</a>.',
// Match for educator CI-H mouseover
'icon-question-cih.png': '<strong>CI-H</strong> (Communication Intensive in the Humanities, Arts, and Social Sciences): MIT undergraduates must take two such courses, which provide a foundation in writing and oral communication. <a href="http://web.mit.edu/commreq/cih.html">Learn more</a>.',
// Match for educator CI-HW mouseover						  
'icon-question-cihw.png':'<strong>CI-HW</strong> (Communication Intensive in the Humanities, Arts, and Social Sciences - Writing Focused): A subset of MIT freshmen must take one such course, which focuses on the writing process and rhetoric. <a href="http://web.mit.edu/commreq/faculty%20ci-h.html#CI-HW%20Subjects">Learn more</a>.',
// Match for educator CI-M mouseover						  
'icon-question-cim.png':'<strong>CI-M</strong> (Communication Intensive in the Major): MIT undergraduates must take two such courses, which focus on forms of communication relevant to a given field. <a href="https://registrar.mit.edu/registration-academics/academic-requirements/communication-requirement">Learn more</a>.',
// Match for educator Freshman Advising Seminar mouseover						  
'icon-question-fas.png':'<strong>Freshman Advising Seminar</strong>: A fall-semester, small group, academic class for freshmen that combines advising and learning. <a href="http://mit.edu/firstyear/fas/">Learn more</a>.',
// Match for educator H-Level Graduate Credit mouseover						  
'icon-question-glevel.png':'<strong>G-Level Graduate Credit</strong>: Courses with this designation are approved for graduate credit.',
// Match for educator GIR mouseover						  
'icon-question-gir.png':'<strong>GIR</strong> (General Institute Requirements): MIT undergraduates must take certain courses in science and technology; communication; humanities, arts, and social sciences; laboratory; and physical education. <a href="http://catalog.mit.edu/mit/undergraduate-education/general-institute-requirements">Learn more</a>.',
// Match for educator H-Level Graduate Credit mouseover						  
'icon-question-hlevel.png':'<strong>H-Level Graduate Credit</strong>: Courses with this designation are approved higher-level graduate subjects required for MIT graduate programs.',
// Match for educator HASS mouseover						  
'icon-question-hass.png':'<strong>HASS</strong> (Humanities, Arts, and Social Sciences): MIT undergraduates must take eight such courses. <a href="http://web.mit.edu/hassreq/">Learn more</a>.',
// Match for educator HASS-A mouseover				  
'icon-question-hass-a.png':'<strong>HASS-A</strong>: MIT undergraduates must take one course designated as Arts in fulfilling the eight-course requirement in Humanities, Arts, and Social Sciences. <a href="http://web.mit.edu/hassreq/index.html#hasscatdef">Learn more</a>.',
// Match for educator HASS-H mouseover
'icon-question-hass-h.png':'<strong>HASS-H</strong>: MIT undergraduates must take one course designated as Humanities in fulfilling the eight-course requirement in Humanities, Arts, and Social Sciences. <a href="https://registrar.mit.edu/registration-academics/academic-requirements/hass-requirement#hasscatdef">Learn more</a>.',
// Match for educator HASS-S mouseover
'icon-question-hass-s.png':'<strong>HASS-S</strong>: MIT undergraduates must take one course designated as Social Sciences in fulfilling the eight-course requirement in Humanities, Arts, and Social Sciences. <a href="http://web.mit.edu/hassreq/index.html#hasscatdef">Learn more</a>.',
// Match for educator IAP mouseover
'icon-question-iap.png':'<strong>IAP</strong> (Independent Activities Period): MIT runs a special four-week term between MIT\'s Fall and Spring semesters. <a href="http://web.mit.edu/iap/">Learn more</a>.',
// Match for educator Lab mouseover
'icon-question-lab.png':'<strong>Lab</strong>: MIT undergraduates must take one such course, which emphasizes experimental problems and introduces students to how research in the discipline is conducted. <a href="http://catalog.mit.edu/mit/undergraduate-education/general-institute-requirements/#laboratoryrequirementtext">Learn more</a>.',
// Match for educator REST mouseover
'icon-question-rest.png':'<strong>REST</strong> (Restricted Electives in Science and Technology): MIT undergraduates must take two such courses, which broaden and deepen students\' understanding of basic science and scientific inquiry. <a href="http://catalog.mit.edu/mit/undergraduate-education/general-institute-requirements/#restrequirementtext">Learn more</a>.',
// Match for educator UROP mouseover
'icon-question-urop.png':'<strong>UROP</strong> (Undergraduate Research Opportunities Program): This program provides MIT undergraduates with the resources to pursue research projects under supervision of MIT faculty. <a href="http://web.mit.edu/urop/">Learn more</a>.',
// Match for educator term bubble #1 mouseover - active learning
'icon-question-active.png':'<strong>Active Learning</strong>: See <a href="http://files.eric.ed.gov/fulltext/ED336049.pdf">Bonwell &amp; Eison (1991)</a>.',
// Match for educator term bubble #2 mouseover - clicker questions				  
'icon-question-clickq.png':'<strong>Clicker Questions</strong>: See <a href="http://www.amazon.com/Eric-Mazur-Instruction-Users-Manual/dp/B008UYZT4A/ref=nosim/mitopencourse-20">Mazur (1997)</a> and <a href="http://www.sciencemag.org/content/323/5910/50">Mazur (2009)</a>.',
// Match for educator term bubble #3 mouseover - concept questions
'icon-question-conq.png':'<strong>Concept Questions</strong>: See <a href="http://www.amazon.com/Eric-Mazur-Instruction-Users-Manual/dp/B008UYZT4A/ref=nosim/mitopencourse-20">Mazur (1997)</a> and <a href="http://mazur.harvard.edu/publications.php?function=display&rowid=537">Crouch, Watkins, Fagen &amp; Mazur (2007)</a>.',
// Match for educator term bubble #4 mouseover - culturally relevant pedagogy
'icon-question-ped.png':'<strong>Culturally Relevant Pedagogy</strong>: See <a http://www.tandfonline.com/doi/pdf/10.1080/00405849509543675#.VNosV2TF-4Q">Ladson-Billings (1995)</a>.',
// Match for educator term bubble #5 mouseover - flipped classroom
'icon-question-flip.png':'<strong>Flipped Classroom</strong>: See <a href="http://www.learningandleading-digital.com/learning_leading/200812#pg24">Bergmann &amp; Sams (2008)</a> and <a href="http://www.sciencemag.org/content/323/5910/50">Mazur (2009)</a>.',
// Match for educator term bubble #6 mouseover - formative assessment
'icon-question-form.png':'<strong>Formative Assessment</strong>: See <a href="http://pdk.sagepub.com/content/92/1/81.full.pdf+html">Black &amp; Wiliam (1998)</a>.',
// Match for educator term bubble #7 mouseover - immediate feedback
'icon-question-immed.png':'<strong>Immediate Feedback</strong>: See <a href="http://www.ascd.org/Publications/Books/Overview/How-to-Give-Effective-Feedback-to-Your-Students.aspx">Brookhart (2008)</a> and <a href="http://www.amsciepub.com/doi/pdf/10.2466/pr0.2001.88.3.889">Epstein, Epstein &amp; Brosvic (2001)</a>.',
// Match for educator term bubble #8 mouseover - learning and engagement
'icon-question-learn.png':'<strong>Learning and Engagement</strong>: See <a href="http://www.tandfonline.com/doi/abs/10.1080/14703290701602748#preview">Bryson &amp; Hand (2007)</a> and <a href="http://www.tandfonline.com/doi/pdf/10.1080/00091380309604090">Kuh (2010)</a>.',
// Match for educator term bubble #9 mouseover - reading questions
'icon-question-readq.png':'<strong>Reading Questions</strong>: See <a href="http://mazur.harvard.edu/publications.php?function=display&rowid=537">Crouch, Watkins, Fagen &amp; Mazur (2007)</a>.',
// Match for educator term bubble #10 mouseover - service-learning
'icon-question-service.png':'<strong>Serivce Learning</strong>: See <a href="http://onlinelibrary.wiley.com/doi/10.1002/tl.470/pdf">Felten &amp; Clayton (2011)</a>.',
// Match for educator term bubble #11 mouseover - summative assessment
'icon-question-sum.png':'<strong>Summative Assessment</strong>: See <a href="http://www.tandfonline.com/doi/pdf/10.1111/j.1467-8527.2005.00307.x#.VNovpGTF-4Q">Taras (2005)</a>.',
// Match for educator term bubble #12 mouseover - collaborative learning
'icon-question-collab.png':'<strong>Collaborative Learning</strong>: See <a href="www.amazon.com/Collaborative-Learning-Education-Interdependence-Authority/dp/0801859743/ref=nosim/mitopencourse-20">Bruffee (1998)</a>.',
// Match for educator term bubble #13 mouseover - cooperative learning
'icon-question-coop.png':'<strong>Cooperative Learning</strong>: See <a href="http://files.eric.ed.gov/fulltext/ED343465.pdf">Johnson, Johnson &amp; Smith (1991)</a>.',
// Match for other version tab tool tip mouseover
'icon-question-ovt.png':'OCW courses are "snapshots" of a subject as taught in a particular term by specific faculty. Multiple versions in OCW show how the subject has evolved due to changes in the field, different faculty, and new pedagogy.',
// Match for Scholar Version tool tip mouseover
'icon-question-svt.png':'OCW Scholar courses are among OCW\'s most complete courses, blending content created expressly for OCW with material from on-campus MIT classes. <a href="https://ocw.mit.edu/courses/ocw-scholar/">Learn more about OCW Scholar</a>. ',
// Match for Archived Version tool tip mouseover
'icon-question-avt.png':'These older versions have been replaced by more recent versions, and are archived in the <a href="https://dspace.mit.edu/">DSpace@MIT Repository</a>. ',
// Match for MITx BubblePopup	edx-open
'icon-question-edx-open.png':'MITx courses typically have defined start and end dates, interactive exercises, integrated assessment, online forums with course staff interaction, and options for certificates of completion. <a href="https://www.edx.org/school/mitx">Learn more about MITx</a>.',
// Match for MITx BubblePop	edx-archived
'icon-question-edx-archived.png':'Archived MITx courses allow self-study of some course materials and videos after the course has closed. Some features like online forums and certificates may not be available. <a href="https://www.edx.org/school/mitx">Learn more about MITx</a>.',
// Match for educator bubble mouseover - crosslinks
'icon-question-cl.png':'STEM: Science, Technology, Engineering and Math',
// Match for educator bubble mouseover - Unrestricted elective
'icon-question-unrestrict.png':'Unrestricted electives count toward an undergraduate student\’s total credits required for graduation, but do not fulfill specific requirements for any major.',
};
	
var technicalRequirementsMap = {
'.ai' : '<a href="http://www.adobe.com/products/illustrator/whatisillustrator/">Adobe Illustrator</a> must be used to edit these vector graphic files.',
'.aif' : 'Audio Interchange File Format (AIFF). Any media player or sound editor program can be used to view .aif files.',
'.aiff' : 'Audio Interchange File Format (AIFF). Any media player or sound editor program can be used to view .aiff files.',
'.ape' : '.ape file extensions refer to "A plasmid Editor" (ApE), a sequence editing software. Files can be viewed using either <a href="http://www.snapgene.com/products/file_compatibility/ApE/">SnapGene</a> or downloading the free <a href="http://biologylabs.utah.edu/jorgensen/wayned/ape/">ApE software</a>.',
'.asc' : 'Any text editor can be used to view .asc files.',
'.avi' : 'Media player software, such as <a href="http://www.apple.com/quicktime/download/">QuickTime Player</a>, <a href="http://www.real.com/realplayer">Real Media Player</a>, or <a href="https://support.microsoft.com/en-us/help/14209/get-windows-media-player">Windows Media Player</a>, is required to run .avi files.',
'.bat' : 'Any text editor can be used to view .bat files. These files are also executable.',
'.bin' : 'File decompression software, such as Winzip or StuffIt, is required to open .bin files.',
'.bmp' : 'Any number of image viewers can be used to view .bmp files',
'.c' : 'Any number of development tools can be used to compile and run .c files. They can also be opened with any text editor.',
'.cbp' : 'Central Builder Project files are associated with C++ and can be opened with any text editor.',
'.cc' : 'Any number of development tools can be used to compile and run .cc files, which are C++ source code.',
'.cif' : 'A standard filetype sponsored by <a href="http://www.iucr.org/resources/cif">IUCr</a>, used by various <a href="http://www.iucr.org/resources/cif/software">software and tools</a>.',
'.cir' : 'These files can be read using <a href="http://www.cadence.com/products/orcad/pages/default.aspx">PSPICE Schematics</a>, available from Cadence, Inc.',
'.class' : '<a href="http://www.java.com/">Java Virtual Machine</a> (automatically installed in most major Web browsers) is required to run .class files.',
'.cnf' : 'Any text editor can be used to view .cnf files.',
'.cpp' : 'Any number of development tools can be used to compile and run .cpp files. They can also be opened with any text editor.',
'.coor' : 'Any text editor can be used to view .coor files.',
'.csv' : 'Any number of software tools can be used to import .csv files.',
'.ctd' : '<a href="http://www.mathworks.com/products/matlab/">MATLAB software</a> is required to make use .ctd files. NOTE: This is not a unique MATLAB extension. Please verify it is for use in MATLAB before using these statements.',
'.dat' : 'Any number of software tools can be used to import .dat files.',
'.data' : 'Any number of software tools can be used to import .data files.',
'.db' : '.db files are generic database files that store data in a structured format. They can be read by any number of database programs.',
'.dbf' : '<a href="http://www.esri.com/software/arcview/">ArcView</a> is required to open and work with .dbf files, although they can also be opened with Microsoft Excel.',
'.dcd' : 'The <a href="http://www.ks.uiuc.edu/Research/namd/2.6/ug/node13.html">DCD</a> files are single precision binary FORTRAN files. The format is supported by a wide range of analysis and display programs (such as <a href="http://www.ks.uiuc.edu/Research/vmd/">VMD</a> and <a href="http://www.ks.uiuc.edu/Research/namd/">NAMD</a>).',
'.dcm' : 'An image format created by the National Electrical Manufacturers Association (NEMA) as a standard for distributing and viewing medical images, such as MRIs, CT scans, and ultrasound images.',
'.dcr' : '<a href="http://www.macromedia.com/">Shockwave Player</a> is required to run Shockwave media.',
'.dll' : 'Any number of programs can call .dll files. These files are also executable.',
'.dict' : 'Any number of text editors can be used view .dict files.',
'.dmg' : 'Apple Disk Images are used by the Mac OS X operating system. Use the Disk Utility app to open these.',
'.do' : 'The <a href="http://www.stata.com/info/">STATA statistical package</a> is required to view and run .do files.',
'.doc' : '<a href="http://office.microsoft.com/">Microsoft Word</a> is recommended for viewing .doc files. Free <a href="http://www.microsoft.com/downloads/details.aspx?FamilyID=3657CE88-7CFA-457A-9AEC-F4F827F20CAC&amp;DisplayLang=en">Microsoft Word Viewer software</a>, <a href="http://www.openoffice.org/">OpenOffice Writer</a>, and <a href="http://www.abisource.com">AbiWord</a> can also be used to view .doc files.',
'.ds' : '<a href="http://www.pasco.com/datastudio">DataStudio  Software</a> is required to run .ds files.',
'.dta' : 'The <a href="http://www.stata.com/info/">STATA statistical package</a> is required to view and run .dta files.',
'.dvi' : 'These are supporting files used in conjunction with <a href="#tex">TeX</a> documents. No additional software is required if you can already open or edit the primary document file.',
'.dwg' : 'This is a universal file format that can be opened by applications such as Solidworks, Autocad, Illustrator, or other major CAD software.',
'.dxf' : 'These are supporting files that allow computer-aided design (CAD) projects to operate across many different software packages. No additional software is required if you can already open or edit the primary document file.',
'.el' : 'Lisp source code is primarily associated with Emacs Lisp, a dialect of the Lisp programming language used by the GNU Emacs and XEmacs text editors. It can be opened with any text editor.',
'.eps' : '<a href="http://www.cs.wisc.edu/~ghost/">Ghostscript/Ghostview</a>, <a href="http://www.adobe.com/products/photoshop/">Adobe Photoshop</a>, and <a href="http://www.adobe.com/products/illustrator/">Adobe Illustrator</a> are among the tools that can be used to view .eps files.',
'.exe' : '.exe files are executable programs.',
'.fa' : 'Any number of biological sequence comparison software tools can be used to import <a href="http://en.wikipedia.org/wiki/Fasta_format">FASTA formatted sequence</a> (.fa) files.',
'.fas' : 'Any number of biological sequence comparison software tools can be used to import <a href="http://en.wikipedia.org/wiki/Fasta_format">FASTA formatted sequence</a> (.fas) files.',
'.fasta' : 'Any number of biological sequence comparison software tools can be used to import <a href="http://en.wikipedia.org/wiki/Fasta_format">FASTA formatted sequence</a> (.fasta) files.',
'FEFLOW' : '<a href="http://www.wasy.de/english/produkte/feflow/download.html">FEFLOW</a> is required to use these files.',
'.fig' : '<a href="http://www.mathworks.com/products/matlab/">MATLAB software</a> is required to view and run .fig files (MATLAB graphics files).',
'.fin' : '.fin fles are data offset tables and can be opened using any text edior.',
'.fits' : 'Flexible Image Transport System (FITS) is a digital file format used to store, transmit, and manipulate scientific and other images. FITS is the most commonly used digital file format in astronomy. <a href="http://hea-www.harvard.edu/RD/ds9/">DS9</a> is an application that supports these files and is <a href="http://hea-www.harvard.edu/RD/ds9/">available for free</a>.',
'.f90' : 'Any number of development tools can be used to compile and run .f90 FORTRAN 90 files.',
'.for' : 'Any number of development tools can be used to compile and run .for FORTRAN files.',
'.f' : 'Any number of development tools can be used to compile and run .f files.',
'.fortran' : 'Any number of development tools can be used to compile and run .fortran files.',
'.fna' : 'Any number of biological sequence comparison software tools can be used to import <a href="http://en.wikipedia.org/wiki/Fasta_format">FASTA formatted sequence</a> (.fna) files.',
'.fsm' : 'FlexSim Simulation Software is required to open and work with .fsm files. A free trial version of FlexSim is available.',
'.gb' :'.gb file extensions refer here to sequence files, commonly used by <a href="http://www.ncbi.nlm.nih.gov/genbank/">GenBank</a>, a sequence database, for its record files. <a href="http://www.snapgene.com/products/file_compatibility/GenBank/">Snap Gene and Snap Gene Viewer</a> can be downloaded for free to view these files.',
'.gff' : 'GFF is a format for describing genes and other features associated with DNA, RNA and protein sequences. The current specification and several software applications are available from <a href="http://www.sanger.ac.uk/resources/software/">the Sanger Institute</a>.',
'.gblorb' : 'These compiled versions of <a href="#ni">.ni</a> files can be run with an interpreter such as <a href="http://frotz.sourceforge.net/">Frotz</a>.',
'.gph' : 'The <a href="http://www.stata.com/info/">STATA statistical package</a> is required to view and run .gph files.',
'.gz' : 'File decompression software, such as <a href="http://www.winzip.com/">Winzip</a> or <a href="http://www.stuffit.com/">StuffIt</a>, is required to open .gz files.',
'.h' : 'Header files for .c, .cc, or .cpp programs. Any number of development tools can be used to compile and run the C/C++ source files that reference .h files.',
'.hs' : '<a href="http://getabest.net/glasgow-haskell-compiler-download-new-120391.html">Haskell compiler software</a> is required to run .hs files.',
'.idb' : 'This filetype is associated with <a href="http://www.adina.com/index.shtml">ADINA</a>, the FEA modeling software developed by Prof. Klaus-J&uuml;rgen Bathe.',
'.igs' : '.igs is a CAD translation standard ASCII text-based format for saving and exporting vector data.',
'.in' : 'This filetype is associated with <a href="http://www.adina.com/index.shtml">ADINA</a>, the FEA modeling software developed by Prof. Klaus-J&uuml;rgen Bathe. Any text editor can be used to view the contents of .in files.',
'.iv' : 'The ivview program from <a href="http://oss.sgi.com/projects/inventor/">SGI Open Inventor</a> toolkit is required to view .iv files.',
'.jar' : '<a href="http://java.sun.com/products/plugin/">Java plug-in software</a> is required to run Java files.',
'.java' : 'Any number of development tools can be used to compile and run .java files.',
'.jmp' : '<a href="http://www.jmp.com/">JMP Statistical Discovery Software</a> is required to run JMP files.',
'.jnlp' : '<a href="http://java.sun.com/products/javawebstart/">Java Web Start software</a> is required to run Java files.',
'.jsim' : 'JSim is a computer-aided design tool that provides a simple editor for entering a circuit description, programs to simulate circuits, and a waveform browser to view the results of a simulation.',
'.key' : '<a href="http://www.apple.com/iwork/keynote/">Keynote</a>, a component of <a href="http://www.apple.com/iwork/">iWork</a>, is required for viewing .key files.',
'.kml' : 'Keyhole Markup Language (KML) is an XML-based language schema for expressing geographic annotation and visualization on existing or future Web-based, two-dimensional maps and three-dimensional Earth browsers.',
'.lad' : '<a href="http://www.mathworks.com/products/matlab/">MATLAB software</a> is required to use .lad files. NOTE: This is not a unique MATLAB extension. Please verify it is for use in MATLAB before using these statements.',
'.ldf' : '<a href="http://www.microsoft.com/sqlserver/">Microsoft SQL Server</a> is required to use .ldf files. Free <a href="http://www.microsoft.com/express/Database/">Microsoft SQL Server Express</a> can also be used with .ldf files.',
'.lib' : 'These files can be read using <a href="http://www.cadence.com/products/orcad/pages/default.aspx">PSPICE Schematics</a>, available from Cadence, Inc.',
'.lisp' : '<a href="http://www.franz.com/products/allegrocl/">Allegro Common Lisp</a> software is required to run .lisp files.',
'.loc' : '<a href="http://www.mathworks.com/products/matlab/">MATLAB software</a> is required to use .loc files. NOTE: This is not a unique MATLAB extension. Please verify it is for use in MATLAB before using these statements.',
'.m' : '<a href="http://www.mathworks.com/products/matlab/">MATLAB software</a> is required to run .m files. Additional toolbox plug-ins may also be required. Any text editor can be used to view the contents of .m files. <a href="http://www.gnu.org/software/octave/">GNU Octave</a> can also be used to run some .m files.',
'.mat' : '<a href="http://www.mathworks.com/products/matlab/">MATLAB software</a> is required to view and run .mat files.',
'.mcd' : '<a href="http://www.mathcad.com/">Mathcad software</a> is required to view and run .mcd files.',
'.mdb' : '<a href="http://office.microsoft.com/">Microsoft Access</a> is recommended for viewing .mdb files. Free <a href="http://office.microsoft.com/en-us/downloads/office-online-file-converters-and-viewers-HA001044981.aspx">Microsoft Access viewer</a> can also be used to view .mdb files.',
'.mdf' : '<a href="http://www.microsoft.com/sqlserver/">Microsoft SQL Server</a> is required to use .mdf files. Free <a href="http://www.microsoft.com/express/Database/">Microsoft SQL Server Express</a> can also be used with .mdf files.',
'.mdl' : '.mdl files are Mathworks <a href="http://www.mathworks.com/products/simulink/">Simulink</a>&reg; models, which can be used with a number of software packages, including <a href="http://www.mathworks.com/products/matlab/">MATLAB</a>&reg; and <a href="http://vensim.com/vensim-personal-learning-edition/">Vensim PLE</a>&reg; (free for academic use).',
'.mf' : 'These are metafont text or source files. To use these files, you need a typesetting program that can read the font such as <a href="http://tug.ctan.org/">TeX</a> or Metafont.',
'.mid' : 'TMedia player software, such as <a href="http://www.apple.com/quicktime/download/">QuickTime Player</a>, <a href="http://www.real.com/realplayer">Real Media Player</a>, or <a href="https://support.microsoft.com/en-us/help/14209/get-windows-media-player">Windows Media Player</a> is required to run MID files.',
'.midi' : 'TMedia player software, such as <a href="http://www.apple.com/quicktime/download/">QuickTime Player</a>, <a href="http://www.real.com/realplayer">Real Media Player</a>, or <a href="https://support.microsoft.com/en-us/help/14209/get-windows-media-player">Windows Media Player</a> is required to run MIDI files.',
'.mnc' : 'MINC is a medical imaging oriented extension of the NetCDF file format. NetCDF stands for &quot;Network Common Data Form&quot; and it is widely used in many areas of scientific data storage.',
'.mod' : 'Any number of programs can be used to run .mod files.',
'.mov' : '<a href="http://www.apple.com/quicktime/download/">QuickTime Player software</a> is required to view .mov files.',
'.movie' : '<a href="http://www.apple.com/quicktime/download/">QuickTime Player software</a> is required to view .movie files.',
'.mp' : '<a href="http://www.tug.org/metapost.html">MetaPost</a> is a picture-drawing language that outputs Encapsulated PostScript (<a href="#eps">.eps</a>) files.',
'.mpp' : 'Project management software, such as <a href="http://office.microsoft.com/en-us/project/FX100487771033.aspx">Microsoft Project</a> is required to use these files.',
'.mpeg' : 'Media player software, such as <a href="http://www.apple.com/quicktime/download/">QuickTime Player</a>, <a href="http://www.real.com/realplayer">Real Media Player</a>, or <a href="https://support.microsoft.com/en-us/help/14209/get-windows-media-player">Windows Media Player</a> is required to view these files.',
'.mpg' : 'Media player software, such as <a href="http://www.apple.com/quicktime/download/">QuickTime Player</a>, <a href="http://www.real.com/realplayer">Real Media Player</a>, or <a href="https://support.microsoft.com/en-us/help/14209/get-windows-media-player">Windows Media Player</a> is required to view these files.',
'.mph' : '<a href="http://www.femlab.com/products/multiphysics/">COMSOL MULTIPHYSICS</a>(formerly FEMLAB) is required to use .mph files.',
'.mpp' : 'Project management software, such as <a href="http://office.microsoft.com/en-us/project/FX100487771033.aspx">Microsoft Project</a> is required to use these files.',
'.msi' : 'Microsoft Installer files are software components used for installation, maintenance, removal of software on Microsoft Windows systems.',
'.mw' : 'These files are Maple Worksheet files, for use with <a href="http://www.maplesoft.com/products/maple/">Maple</a> software.',
'.nb' : '<a href="http://www.wolfram.com/products/mathematica/index.html">Mathematica software</a> or the free <a href="http://www.wolfram.com/products/mathreader/">MathReader</a> software is required to run .nb files.',
'.net' : '.net files describe a network. <a href="http://pajek.imfm.si/doku.php">Pajek</a> is required to analyze and visualize .net files.',
'.ni' : 'Structured text .ni can be read and edited with WordPad, Microsoft Word, or any other text editor. They are designed for use with <a href="http://inform7.com/">Inform 7</a>, a design system for interactive fiction.',
'OASES' : 'OASES is a general purpose computer code for modeling seismo-acoustic propagation in horizontally stratified waveguides using wavenumber integration in combination with the Direct Global Matrix solution technique. It works with a variety of file types, including .par, .bdr, .cdr, .plp, .plt, .mtv, .trf and .rco.',
'.ods' : '.ods files are OpenDocument Spreadsheet files and can be opened using OpenOffice Calc, Microsoft Excel, or most other spreadsheet programs.',
'.ord' : '<a href="http://www.omax.com/">OMAX&#194; Software</a> is required to run .ord files.',
'.oum' : 'Any text editor can be used to view the contents of these .oum files. To use the .oum file as instructed, <a href="http://www.mathworks.com/products/matlab/">MATLAB software</a> is required.',
'.out' : '.out fles are data offset tables and can be opened using any text edior.',
'.ova' : '.ova files contain a Virtual Machine drive image and can be opened with <a href="https://www.virtualbox.org/">Virtual Box</a> or similar software.',
'.p' : '<a href="http://www.mathworks.com/products/matlab/">MATLAB software</a> is required to run the .p files found on a course site. Unlike .m files, these are binary runtime files, and cannot be opened with a text editor.',
'.p3' : '<a href="http://www.primavera.com/products/index.asp">Primavera Project Planner</a> software is required to run .p3 files on a course site.',
'.pd' : 'These are files for use in <a href="http://puredata.info/">Pure Data</a>, an open source real-time graphical programming environment for audio, video, and graphical processing.',
'.pdb' : '<a href="http://www.openrasmol.org/">RasMol software</a> is required to run .pdb files.',
'.pdb' : '<a href="http://www.rcsb.org/robohelp_f/#molecular_viewers/introduction_to_molecular_viewers.htm">Protein Data Bank</a> viewer software can be used to view .pdb protein data bank files.',
'.pde' : 'These files are associated with <a href="http://processing.org/">Processing</a>, a programming language and environment for people who want to program images, animation and interactions.',
'.pddl' : 'Any text editor can be used to view .pddl files.',
'.pl' : 'A <a href="http://www.perl.org/get.html">Perl interpreter</a> is required to run .pl files.',
'.ppt' : '<a href="http://office.microsoft.com/">Microsoft PowerPoint</a> is recommended for viewing .ppt files. Free <a href="http://office.microsoft.com/en-us/downloads/office-online-file-converters-and-viewers-HA001044981.aspx">Microsoft PowerPoint viewer</a> can also be used to view .ppt files.',
'.pptx' : '<a href="http://office.microsoft.com/">Microsoft PowerPoint</a> is recommended for viewing .pptx files. Free <a href="http://office.microsoft.com/en-us/downloads/office-online-file-converters-and-viewers-HA001044981.aspx">Microsoft PowerPoint viewer</a> can also be used to view .pptx files.',
'.ps' : 'Postscript viewer software, such as <a href="http://www.cs.wisc.edu/~ghost/">Ghostscript/Ghostview</a>, can be used to view .ps files.',
'.psd' : '.psd files are images that can be viewed and edited with <a href="http://www.adobe.com/products/photoshop.html">Adobe Photoshop</a>. Lesser functionality with .psd files is available through the free programs <a href="http://www.gimp.org/">GIMP</a> and <a href="http://psdviewer.org/">PSD Viewer</a>.',
'.psf' : 'Any text editor can view these Protein Structure (.psf) files. They are generated by <a href="http://www.ks.uiuc.edu/Research/vmd/">VMD Molecular Graphics Viewer</a> and can be used with <a href="http://www.ks.uiuc.edu/Research/namd/">NAMD (Molecular Dynamics Simulator)</a>.',
'.py' : 'Use the <a href="http://www.python.org/">Python interpreter</a> to run .py files.',
'.qif' : '<a href="http://www.mathworks.com/products/matlab/">MATLAB software</a> is required to run .qif files. Any text editor can be used to view the contents of .qif files.',
'.r' : 'Use <a href="http://www.r-project.org/">R</a>, an open source language and development environment for statistical computing, graphics, and exploratory data analysis, to run .r files.',
'.ram' : '<a href="http://www.real.com/realplayer">Real Media Player</a> is required to run .ram files.',
'.rar' : 'Files compressed using the .rar format can be extracted with the same software used to handle .zip (e.g., WinZip and StuffIt).',
'.rb' : 'A <a href="http://www.ruby-lang.org/en/">Ruby interpreter</a> is required to run .rb files.',
'.rm' : '<a href="http://www.real.com/realplayer">Real Media Player software</a> is required to run .rm files.',
'.rpt' : '.rpt is a report file. Generally, these are ASCII files and can be read using any text editor.',
'.rt' : '<a href="http://www.real.com/realplayer">Real Media Player</a> is required to run .rt files.',
'.rtf' : 'Rich Text Format files can be read with any word-processing program such as Microsoft WordPad, Apple TextEdit, <a href="http://www.libreoffice.org/">LibreOffice</a>, <a href="http://www.openoffice.org/">OpenOffice</a>.',
'.s' : 'Assembly file. Any number of development tools can be used to compile and run .s files.',
'.sas7bdat' : '<a href="http://www.sas.com/technologies/bi/appdev/base/">Base SAS software</a> is required to open .sas7bdat files.',
'.scm' : '<a href="http://www.gnu.org/software/mit-scheme/">Scheme software</a> is required to run .scm files.',
'.scs' : '<a href="http://www.cadence.com/products/rf/spectre_circuit/pages/default.aspx">Virtuoso Spectre</a> software is required to run .scs files.',
'.sit' : '<a href="http://www.stuffit.com/">Stuffit</a> is required to open .sit files.',
'.slb' : 'These files can be read using <a href="http://www.cadence.com/products/orcad/pages/default.aspx">PSPICE Schematics</a>, available from Cadence, Inc.',
'.sldasm' : '<a href="http://www.solidworks.com/pages/products/products.html">SolidWorks software</a> is required to run and view .sldasm files.',
'.sldprt' : '<a href="http://www.solidworks.com/pages/products/products.html">SolidWorks software</a> is required to run and view .sldprt files.',
'.slddrw' : '<a href="http://www.solidworks.com/pages/products/products.html">SolidWorks software</a> is required to run and view .slddrw files.',
'.slogo' : '<a href="http://education.mit.edu/starlogo">StarLogo software</a> is required to run .slogo files.',
'.smil' : '<a href="http://www.real.com/realplayer">Real Media Player</a> is required to run .smil files.',
'.so' : 'These are Unix shared library files. Any number of programs can call .so files. These files are also executable.',
'.spt' : '<a href="http://www.openrasmol.org/">RasMol software</a> is required to run .spt files.',
'.sql' : 'These data files for SQL (Structured Query Language) can be edited using any text editor, and read by any SQL-compatible database program, including <a href="http://www.filemaker.com/">FileMaker</a>, <a href="http://office.microsoft.com/">Microsoft Access</a> and <a href="http://www.mysql.com/">MySQL</a>.',
'.stl' : 'Computer-aided design (CAD) software is required to view .stl files.',
'.stp' : '<a href="http://www.nanotec.es/products/wsxm/download.php">Nanotec WSxM software</a> is required to view .stp file.',
'.sty' : 'These are supporting files used in conjunction with <a href="#tex">TeX</a> documents. No additional software is required if you can already open or edit the primary document file.',
'.sub' : 'Subcircuit files used in the <a href="http://www.linear.com/designtools/software/ltspice.jsp">LTSpice graphical SPICE simulator</a>.',
'.svn' : '.svn files are used by <a href="http://subversion.apache.org/">Subversion</a>, an open source version control system.',
'.swf' : '<a href="http://www.macromedia.com/">Flash Player</a> is required to run Flash media.',
'.swj' : '<a href="http://www.solidworks.com/pages/products/products.html">SolidWorks software</a> is required to run and view .swj files.',
'.sws' : '<a href="http://www.sagemath.org">Sage</a> is a free open-source mathematics software system licensed under the GPL and required to run the .sws files on our site.',
'.tar' : 'File decompression software, such as <a href="http://www.winzip.com/">Winzip</a> or <a href="http://www.stuffit.com/">StuffIt</a>, is required to open .tar files.',
'.tex' : 'Software to view .tex files is available from the <a href="http://www.ctan.org/">Comprehensive TeX Archive Network (CTAN)</a> and the <a href="http://www.tug.org/">TeX Users Group Web site</a>.',
'.textgrid' : '<a href="http://www.fon.hum.uva.nl/praat/">Pratt</a> software is needed to use these plain text files that accompany corresponding sound files.',
'.tgz' : 'File decompression software, such as Winzip or StuffIt, is required to open .tgz files.',
'.tif' : 'Any number of image viewers can be used to view .tif files.',
'.uasm' : 'A &quot;micro&quot; assembly language used in course 6.004. The files in this course contain all information required to use them.',
'.ucf' : 'Any number of text editors can be used to open and modify .ucf files. In many cases, they can also be used in an FPGA design and testing environment such as <a href="http://www.xilinx.com/ise/logic_design_prod/webpack.htm">Xilinx ISE WebPACK</a>.',
'.v' : 'Any number of text editors can be used to open and modify .v files. In many cases, they can also be used in an FPGA design and testing environment such as <a href="http://www.xilinx.com/ise/logic_design_prod/webpack.htm">Xilinx ISE WebPACK</a>.',
'.vdf' : 'These files are associated with <a href="http://www.vensim.com/software.html">Vensim Simulation Software</a>, which is used for developing high-quality dynamic feedback models.',
'.vpp' : 'These files are associated with Visual Paradigm, a series of software development applications. Product downloads are available from the <a href="http://www.visual-paradigm.com/download/">Visual Paradigm website</a>.',
'.vsd' : '<a href="http://office.microsoft.com/">Microsoft Visio</a> is recommended for viewing .vsd files. Free <a href="http://office.microsoft.com/en-us/downloads/office-online-file-converters-and-viewers-HA001044981.aspx">Microsoft Visio viewer</a> can also be used to view .vsd files.',
'.vss' : '<a href="http://office.microsoft.com/">Microsoft Visio</a> is recommended for viewing .vss stencil files. Free <a href="http://office.microsoft.com/en-us/downloads/office-online-file-converters-and-viewers-HA001044981.aspx">Microsoft Visio viewer</a> can also be used to view .vss files.',
'.vst' : '<a href="http://office.microsoft.com/">Microsoft Visio software</a> is recommended for viewing .vst template files. Free <a href="http://office.microsoft.com/en-us/downloads/office-online-file-converters-and-viewers-HA001044981.aspx">Microsoft Visio viewer</a> can also be used to view .vst files.',
'.wav' : 'Media player software, such as <a href="http://www.apple.com/quicktime/download/">QuickTime Player</a>, <a href="http://www.real.com/realplayer">Real Media Player</a>, or <a href="https://support.microsoft.com/en-us/help/14209/get-windows-media-player">Windows Media Player</a> is required to run .wav files.',
'.wscp' : 'Any text editor can be used to view .wcsp files.',
'.wk1' : 'A Lotus123 worksheet that can be read directly into <a href="http://www.mathworks.com/products/matlab/">MATLAB</a>. <a href="http://www-142.ibm.com/software/sw-lotus/products/product2.nsf/wdocs/123home">Lotus 123</a> or <a href="http://www.mathworks.com/products/matlab/">MATLAB software</a> can be used to run .wk1 files.',
'.wm' : '<a href="http://www.design-simulation.com/WM2D/index.php">Working Model 2D</a> is required to view and run .wm files.',
'.wma' : 'Media player software, such as <a href="http://www.real.com/realplayer">Real Media Player</a>, or <a href="https://support.microsoft.com/en-us/help/14209/get-windows-media-player">Windows Media Player</a> is required to run .wma files.',
'.wmv' : 'Media player software, such as <a href="http://www.real.com/realplayer">Real Media Player</a>, or <a href="https://support.microsoft.com/en-us/help/14209/get-windows-media-player">Windows Media Player</a> is required to run .wmv files.',
'.x' : 'Any number of development tools can be used to run .x files.',
'.xlo' : '.xlo is an <a href="http://usa.autodesk.com/adsk/servlet/pc/index?siteID=123112&amp;id=13717655">AutoDesk</a> Export Journal.',
'.xls' : '<a href="http://office.microsoft.com/">Microsoft Excel</a> is recommended for viewing the .xls files. The free <a href="http://office.microsoft.com/en-us/downloads/office-online-file-converters-and-viewers-HA001044981.aspx">Microsoft Excel viewer</a> can also be used to view .xls files.',
'.xlsx' : '<a href="http://office.microsoft.com/">Microsoft Excel</a> is recommended for viewing the .xlsx files. The free <a href="http://office.microsoft.com/en-us/downloads/office-online-file-converters-and-viewers-HA001044981.aspx">Microsoft Excel viewer</a> can also be used to view .xlsx files.',
'.xmcd' : '<a href="http://www.mathcad.com/">Mathcad software</a> is required to view and run .xmcd files.',
'.xml' : 'Any text editor can be used to view .xml files.',
'.xsl' : 'Any text editor can be used to view .xsl files.',
'.xyz' : '<a href="http://www.ks.uiuc.edu/Research/vmd/">VMD (Molecular Graphics Viewer)</a> is used to to view and use .xyz files.',
'.yaml' : '<a href="http://www.yaml.org/">YAML</a> files organize data in a human-readable text file, and may be viewed or edited using any text editor.',
'.zip' : 'Most modern operating systems will decompress .zip files. If yours does not, try file decompression software, such as <a href="http://www.pkware.com/software/pkzip/windows/">PKZIP</a>, <a href="http://www.winzip.com/">Winzip</a>, or <a href="http://www.stuffit.com/">StuffIt</a>.',
};
    
   $.each( bubblePropertiesMap, function( img, innerHtml) {
		$('img[src$="'+img+'"]').each(function(){		
			setBubblePopupProprties($(this),innerHtml);
		});		
	});
 	
    // Match for Technical requirements
    $.each(technicalRequirementsMap, function(key, value) {
        var selectedAnchorTags = new Array();
        $('a').each(function(){
            var aHref = $(this).attr('href');
            if(typeof aHref != 'undefined'){
                var aHrefLowerCase = aHref.toString().toLowerCase();
                // only proceed in the HREF does NOT contain rss.xml or doi.org 
                var indexRSS = aHrefLowerCase.indexOf("rss.xml");
                var indexDOI = aHrefLowerCase.indexOf("doi.org");
                if ((indexRSS <= -1) && (indexDOI <= -1))
                {
                      if(aHrefLowerCase.match(key+"$")==key)  selectedAnchorTags[selectedAnchorTags.length]=this;
                }
            }
        });
                
        $(selectedAnchorTags).each(function(){
        
             $(this).CreateBubblePopup({
                                            position: 'top',
                                            tail: {align: 'middle'},
                                            alwaysVisible: false,
                                            selectable: true,
                                            innerHtml: value, 
                                            innerHtmlStyle: {
                                                                color:'#000000', 
                                                                'text-align':'left'
                                                            },
                                            themeName: 	'grey',
                                            themePath: 	path,
                                            themeMargins: {
                                                            total: '13px',
                                                            difference: '2px'
                                                          }
                                       }); 
        
        });
    });

});
