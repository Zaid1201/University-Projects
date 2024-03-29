function detection = streamingFaceRecognition(nOfEach,pauseval,refreshOption)
% streamingFaceRecognition(nOfEach,pauseval)
%
% Demonstrates live acquisition, detection, training, and recognition of
% faces.
%
% �	Initializes a webcam and starts a streaming preview of the audience;
% �	automatically detects faces;
% �	captures a specified number of validated face images
%   (�validated� in the QC sense of the word�is the face image good?);
% �	writes them to an auto-created, labeled subdirectory;
% �	asks (in a while loop) if you want to capture another face;
% �	prompts one to (optionally) provide names for the faces;
% �	automatically and quickly trains a recognizer.
% �	Continue to point the camera at the audience, and detected faces will
%   be AUTOMATICALLY LABELED!!!
%
% TO USE:
% Simply point the camera at at least two people, holding steady until
% you get an indication that the capture of that person is complete....
% When you indicate that you don't want to capture another, the validation,
% training, and recognition will commence automatically!
%
% Brett Shoelson, PhD
% brett.shoelson@mathworks.com
% 3/3/2015

% Modifications:
% 4/19/2015: Return if no selection made. Also, modified the call to
% createMontage to make it more efficient; no padding, and imresize to
% targetSize of 100. Should be faster, probably more robust.
%
% 9/22/2015: Changed default nOfEach to 8, and implemented procedure for
% rejecting bad images. Modified the order of preprocessing steps. Also,
% reorganized code for readability. Better, stronger, faster!

% Copyright The MathWorks, Inc. 2015
% Requires the Computer Vision System Toolbox

try
	vidObj = webcam; %Default; You can tweak webcam properties
	%vidObj.Resolution = ('1280x1024');
catch
	beep;
	disp('Please make sure that a properly recognized webcam is connected and try again.');
	return
end

%%% DEMO SETUP:
detection = 0;
%%% PREPROCESSING OPTIONS:
preprocessOpts.matchHistograms = true;
preprocessOpts.adjustHistograms = false;
preprocessOpts.targetForHistogramAndResize = ...
	imread('targetFaceHistogram.pgm');
preprocessOpts.targetSize = 100;

%%% DIRECTORY MANAGEMENT:
targetDirectory = fullfile(fileparts(which(mfilename)),'AutoCapturedFaces');
validateCapturedImages = true;
personNumber = 1;
bestGuess = 0;
% dirExists = exist(targetDirectory,'dir') == 7;
% if dirExists
% 	prompt = sprintf('Would you like to:\n\nSTART OVER (Clears Existing Data!!)\nAdd Face(s) to recognition set\nor Use recognition set as is?');
% 	refresh = questdlg(prompt,'Face Recognition Options','START OVER','Add Face(s)','Use as is','START OVER');
% 	refreshOption = find(ismember({'START OVER','Add Face(s)','Use as is'},refresh));
% else
% 	mkdir(targetDirectory);
% 	refreshOption = 1;
% end

if refreshOption == 1
    %%
    answer = questdlg('Face the camera and press the Start the button when ready',...
        'Database reset: Adding Faces',...
        'Start','Cancel','Start');
    if answer == "Cancel"
        return
    end
	rmdir(targetDirectory,'s');
	mkdir(targetDirectory)
	mkdir(fullfile(targetDirectory,filesep,['Person' num2str(1)]))
	personNumber = 1;
elseif refreshOption == 2
    answer = questdlg('Face the camera and press the Start the button when ready',...
        'Adding Faces to database',...
        'Start','Cancel','Start');
    if answer == "Cancel"
        return
    end
	tmp = dir(targetDirectory);
	fcn = @(x)ismember(x,{'.','..'});
	tmp = cellfun(fcn,{tmp.name},'UniformOutput',false);
	personNumber = nnz(~[tmp{:}])+1;
	mkdir(fullfile(targetDirectory,filesep,['Person' num2str(personNumber)]))

    
elseif refreshOption == 3
    disp("Facial Recognition Triggered")
	% Use as is--no validation of capture!
	validateCapturedImages = false;
elseif isempty(refreshOption)
	delete(vidObj)
	return
end

%if refreshOption == 1 || refreshOption == 2
    %%% FIGURE
    fdrFig = figure('windowstyle','normal',...
	    'name','RECORDING...',...
	    'units','normalized',...
	    'menubar','none',...
	    'position',[0.2 0.1 0.6 0.7]);
%         ,...
% 	    'closerequestfcn',[],...
% 	    'currentcharacter','0',...
% 	    'keypressfcn',@checkForEscape);
    
    %%% Quality Control Options
    %DETECTORS: for upright faces; and for QE, Nose and Mouth
    % Note: these seem to be unnecessary, and to cause capture problems.
    QC.oneNose = false;
    QC.oneMouth = false;
    if QC.oneNose
	    QC.noseDetector = vision.CascadeObjectDetector(...
		    'ClassificationModel','Nose','MergeThreshold',10);
    end
    if QC.oneMouth
	    QC.mouthDetector = vision.CascadeObjectDetector(...
		    'ClassificationModel','Mouth','MergeThreshold',10);
    end
    % H,W of bounding box must be at least this size for a proper detection
    QC.minBBSize = 30; 
    
    % Create face detector
    faceDetector = vision.CascadeObjectDetector('MergeThreshold',10);
    
    % Number of images of each person to capture:
    if nargin < 1
	    nOfEach = 8;
    end
    %Between captured frames (allow time for movement/change):
    if nargin < 2
	    pauseval = 0.5;
    end
    % For cropping of captured faces:
    bboxPad = 25;
    %
    captureNumber = 0;
    isDone = false;
    getAnother = true;

    %%% START: Auto-capture/detect/train!!!
    RGBFrame = snapshot(vidObj);
    frameSize = size(RGBFrame);
    imgAx = axes('parent',fdrFig,...
	    'units','normalized',...
	    'position',[0.05 0.45 0.9 0.45]);
    imgHndl = imshow(RGBFrame);shg;
    
    if ismember(refreshOption,[1,2]) && getAnother && ~isDone
	    while getAnother % && double(get(fdrFig,'currentCharacter')) ~= 27
		    % If successful, displayFrame will contain the detection box.
		    % Otherwise not.
		    [displayFrame, success] = capturePreprocessDetectValidateSave;
		    if success
			    captureNumber = captureNumber + 1;
		    end
		    set(imgHndl,'CData',displayFrame);
		    if captureNumber >= nOfEach
			    beep;pause(0.25);beep;
			    queryForNext;
		    end
	    end %while getAnother
    end
    
    %%% Capture is done. Now for TRAINING:
    imgSet = imageSet(targetDirectory,'recursive');
    if numel(imgSet) < 2
        close(fdrFig)
	    disp('You must capture at least two individuals for this to work!');
        return
    end
    if refreshOption ~= 3
	    queryForNames;
    end
    if validateCapturedImages
	    validateCaptured(imgSet);
        clc
        disp('Images added succesfully.');
        pause(2);
    end
    sceneFeatures = trainStackedFaceDetector(imgSet);
    if refreshOption == 1 || refreshOption == 2
        close(fdrFig)
    end
if refreshOption == 3
%%% Okay, so now we should have a recognizer in place!!!
    time = 0;
    detection = 0;
    while time < 5  %Face recognition returns for 5 seconds
        tic
	    %bestGuess = '?';
	    RGBFrame = snapshot(vidObj);
	    grayFrame = rgb2gray(RGBFrame);
	    bboxes = faceDetector.step(grayFrame);
	    for jj = 1:size(bboxes,1)%#ok
		    if all(bboxes(jj,3:4) >= QC.minBBSize)
			    thisFace = imcrop(grayFrame,bboxes(jj,:));
			    if preprocessOpts.matchHistograms
				    thisFace = imhistmatch(thisFace,...
					    preprocessOpts.targetForHistogramAndResize);
			    end
			    if preprocessOpts.adjustHistograms
				    thisFace = histeq(thisFace);
			    end
			    thisFace = imresize(thisFace,...
				    size(preprocessOpts.targetForHistogramAndResize));
			    %tic;
			    bestGuess = myPrediction(thisFace,sceneFeatures,numel(imgSet));
			    if bestGuess == 0
                    bestGuess = '?';
                    detection = 0;
                else
				    bestGuess = imgSet(bestGuess).Description;
                    RGBFrame = insertObjectAnnotation(RGBFrame, 'rectangle', bboxes(jj,:), bestGuess,'FontSize',48);
                    fprintf("Greetings, %s\n\n",bestGuess);
                    pause(2);
                    detection = 1;
                    close(fdrFig);
                    return
			    end
			    %tPredict = toc
		    end
        end
        RGBFrame = 	insertObjectAnnotation(RGBFrame, 'rectangle', bboxes(jj,:), bestGuess,'FontSize',48);
	    imshow(RGBFrame,'parent',imgAx);drawnow;
	    %title([bestGuess '?'])
        time = time + toc;
    end %while
%     if bestGuess == '?'
%         detection = 0;
%     else
%         detection = 1;
%     end
    
    %%% Clean up:
    delete(vidObj)
    release(faceDetector)
    close(fdrFig)
end
%%
% NESTED SUBFUNCTIONS

	function [displayFrame, success, imagePath] = ...
			capturePreprocessDetectValidateSave(varargin)
		% CAPTURE
		RGBFrame = snapshot(vidObj);
		% Defaults:
		displayFrame = RGBFrame;
		success = false;
		imagePath = [];
		grayFrame = rgb2gray(RGBFrame);
		% PREPROCESS
		if preprocessOpts.matchHistograms
			grayFrame = imhistmatch(grayFrame,...
				preprocessOpts.targetForHistogramAndResize); %#ok<*UNRCH>
		end
		if preprocessOpts.adjustHistograms
			grayFrame = histeq(grayFrame);
		end
		preprocessOpts.targetSize = 100;
		% DETECT
		bboxes = faceDetector.step(grayFrame);
		% VALIDATE
		if isempty(bboxes)
			return
		end
		if size(bboxes,1) > 1
			%disp('Discarding multiple detections!');
			return
		end
		if any(bboxes(3:4) < QC.minBBSize)
			%disp('Bounding box is too small!');
			return
		end
		% On-the-fly QC!
		if QC.oneMouth
			mouthBox = QC.mouthDetector.step(grayFrame);
			if size(mouthBox,1) ~= 1
				%disp('Detected face failed MOUTH QE, and was discarded.')
				return
			end
		end
		if QC.oneNose
			noseBox = QC.noseDetector.step(grayFrame);
			if size(noseBox,1) ~= 1
				%disp('Detected face failed NOSE QE, and was discarded.')
				return
			end
		end
		% If we made it to here, the capture was successful!
		success = true;
		% Update displayFrame
		displayFrame = insertShape(RGBFrame, 'Rectangle', bboxes,...
			'linewidth',4,'color','cyan');
		% SAVE
		% Write to personN directory
		bboxes = bboxes + [-bboxPad -bboxPad 2*bboxPad 2*bboxPad];
		% Make sure crop region is within image
		bboxes = [max(bboxes(1),1) max(bboxes(2),1) min(frameSize(2),bboxes(3)) min(frameSize(2),bboxes(4))];
		faceImg = imcrop(grayFrame,bboxes);
		minImSize = min(size(faceImg));
		thumbSize = preprocessOpts.targetSize/minImSize;
		faceImg = imresize(faceImg,thumbSize);
		% 		if matchHistograms
		% 			faceImg = imhistmatch(faceImg,targetForHistogramAndResize); %#ok<*UNRCH>
		% 		end
		%Defensive programming, since we're using floating arithmetic
		%and we need to make sure image sizes match exactly:
		sz = size(faceImg);
		if min(sz) > preprocessOpts.targetSize
			faceImg = faceImg(1:preprocessOpts.targetSize,1:preprocessOpts.targetSize);
		elseif min(sz) < preprocessOpts.targetSize
			% Not sure if we can end up here, but being safe:
			faceImg = imresize(faceImg,[preprocessOpts.targetSize,preprocessOpts.targetSize]);
		end
		imagePath = fullfile(targetDirectory,...
			['Person' num2str(personNumber)],filesep,['faceImg' num2str(captureNumber) '.png']);
		imwrite(faceImg,imagePath);
		pause(pauseval)
	end %captureAndSaveFrame

% 	function checkForEscape(varargin)
% 		if double(get(gcf,'currentcharacter'))== 27
% 			isDone = true;
% 		end
% 	end %checkForEscape

	function queryForNames
		prompt = {imgSet.Description};
		dlg_title = 'Specify Names';
		def = prompt;
		renameTo = inputdlg(prompt,dlg_title,1,def);
		subfolders = pathsFromImageSet(imgSet);
		for ii = 1:numel(renameTo)
			subf = subfolders{ii};
			fs = strfind(subf,filesep);
			subf(fs(end)+1:end) = '';
			subf = [subf,renameTo{ii}];%#ok
			if ~isequal(subfolders{ii},subf)
				movefile(subfolders{ii},subf);
			end
		end
		imgSet = imageSet(targetDirectory,'recursive');
	end %queryForNames

	function queryForNext
		beep
		captureAnother = questdlg(['Done capturing images for person ', num2str(personNumber), '. Capture Another?'],...
			'Capture Another?','YES','No','YES');
		if strcmp(captureAnother,'YES')
			personNumber = personNumber + 1;
			captureNumber = 0;
			mkdir(fullfile(targetDirectory,filesep,['Person' num2str(personNumber)]))
		else
			getAnother = false;
		end
	end %queryForNext

	function validateCaptured(imgSet)
		%assignin('base','imgSet',imgSet)
		for ii = 1:numel(imgSet)
			nImages = imgSet(ii).Count;
			nCols = ceil(sqrt(nImages));
			nRows = ceil(sqrt(nImages));
			[hobjpos,hobjdim] = distributeObjects(nCols,0.025,0.95,0.025);
			[vobjpos,vobjdim] = distributeObjects(nRows,0.9,0.2,0.1);
			f = togglefig('Validation',true);
			set(f,'windowstyle','normal')
			drawnow
			btn = gobjects(nImages,1);
			ax = btn;
			for jj = 1:nImages %#ok
				ax(jj) = axes('units','normalized',...
					'position',[hobjpos(rem(jj-1,nCols)+1) vobjpos(ceil(jj/nCols)) hobjdim vobjdim]);
				imshow(imread(imgSet(ii).ImageLocation{jj}));
				if jj == 2
					title(imgSet(ii).Description)
				end
				btn(jj) = uicontrol('style','checkbox',...
					'string','Discard',...
					'units','normalized',...
					'value',0,...
					'userdata',jj,...
					'Position',[hobjpos(rem(jj-1,nCols)+1) vobjpos(ceil(jj/nCols))-0.075 hobjdim 0.075]);
			end
			uicontrol('style','pushbutton',...
				'string','Continue',...
				'units','normalized',...
				'position',[0.025 0.025 0.95 0.1],...
				'callback',@registerSelection);
			uiwait(f)
		end
		
		function registerSelection(varargin)
			togglefig('Validation')
			btnvals = find(cell2mat(get(btn,'value')));
			if ~isempty(btnvals)
				confirmDeletion = questdlg(sprintf('Delete the selected %d image(s) from the collection of %s images?', ...
					numel(btnvals),imgSet(ii).Description),...
					'Confirm Deletion','DELETE','No','DELETE');
				if strcmp(confirmDeletion,'DELETE')
					for kk = 1:numel(btnvals)
						imgSet = removeImageFromImageSet(imgSet,imgSet(ii).ImageLocation{btnvals(kk)});
						% Note: deleting images causes problems with the imageset object
						% delete(imgSet(ii).ImageLocation{btnvals(kk)});
					end
				end
			end
			delete(f);
		end %registerSelection (subfunction of validateCaptured)
	end %validateCaptured

end