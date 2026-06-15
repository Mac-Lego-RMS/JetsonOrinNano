% =========================================================================
% WRO - LIVE Visualisierung (ROS 2)  ->  visualize_live.m
% =========================================================================
% Abonniert die in obstacle_run.py implementierten Topics:
%   /global_pose       (geometry_msgs/PoseStamped)      -> Roboter-Trajektorie
%   /global_obstacles  (visualization_msgs/MarkerArray) -> Hindernisse (rot/gruen)
%
% Das Spielfeld ist 3x3 m. Pose- und Hindernis-Koordinaten kommen bereits
% im globalen Feld-Frame "map" (Einheit: Meter, Bereich 0..3) an.
%
% HINWEIS zu Predictions: /global_obstacles enthaelt NUR lokalisierte
% Hindernisse (is_localized==True). Der Prediction-Zustand wird nicht
% gesendet, daher zeichnen wir die Prediction-Platzhalter (Zone 00/01)
% statisch aus DRIVE_DIRECTION (analog zur Logik in obstacle_run.py).
% =========================================================================

clear; clc; close all;

% -------------------------------------------------------------------------
% KONFIGURATION
% -------------------------------------------------------------------------
ROS_DOMAIN      = '0';        % Muss zur Domain des Roboters passen
FIELD_SIZE      = 3.0;        % Spielfeldkantenlaenge in Metern
BG_IMAGE        = 'fe.jpg';   % Hintergrundbild des Feldes
BG_ALPHA        = 1.0;        % Transparenz Hintergrund (0=unsichtbar .. 1=deckend)
RECEIVE_TIMEOUT = 0.5;        % Sekunden, die auf eine neue Pose gewartet wird
MAX_MISSED      = 20;         % Aufeinanderfolgende Timeouts bis "Verbindung verloren"

DRIVE_DIRECTION = 'links';    % 'links' (CCW) oder 'rechts' (CW) -> fuer Predictions
SMOOTH_WINDOW   = 7;          % Fensterbreite des Moving-Average fuer die Trajektorie
AXIS_LEN        = 0.20;       % Laenge des mitfahrenden Roboter-Koordinatensystems (m)
PARK_GAP        = 0.2225;     % Abstand der beiden Parkwaende (22,25 cm) - FIX!
PARK_WALL_LEN   = 0.20;       % Laenge (Tiefe) der Parkwaende in Metern
PARK_CENTER_LON = 1.125;       % Position der Parkbucht entlang der Start-Geraden (m)

setenv('ROS_DOMAIN_ID', ROS_DOMAIN);

% -------------------------------------------------------------------------
% 1. ROS 2 NODE STARTEN
% -------------------------------------------------------------------------
try
    node = ros2node("/matlab_live_view");
    disp('ROS 2 Node "/matlab_live_view" erfolgreich gestartet.');
catch ME
    error(['Konnte Node nicht starten. Laeuft evtl. schon einer? ' ...
           'Nutze "clear" und versuche es erneut.' newline ...
           'Originalfehler: ' ME.message]);
end

% -------------------------------------------------------------------------
% 2. SUBSCRIBER ERSTELLEN
% -------------------------------------------------------------------------
try
    poseSub  = ros2subscriber(node, "/global_pose", "geometry_msgs/PoseStamped");
    obsSub   = ros2subscriber(node, "/global_obstacles", "visualization_msgs/MarkerArray");
    resetSub = ros2subscriber(node, "/global_reset", "std_msgs/Bool");
    disp('Subscriber fuer /global_pose, /global_obstacles und /global_reset aktiv.');
catch ME
    error(['Konnte Subscriber nicht erstellen. Sind die Topic-Typen korrekt ' ...
           'und ist die ROS 2 Toolbox installiert?' newline ...
           'Originalfehler: ' ME.message]);
end

% -------------------------------------------------------------------------
% 3. FIGURE + HINTERGRUNDBILD AUFBAUEN  (alles in EINER Achse!)
% -------------------------------------------------------------------------
fig = figure('Name', 'WRO Live Map', 'Color', 'w');
ax  = axes('Parent', fig);
hold(ax, 'on');
axis(ax, 'equal');
grid(ax, 'on');
xlim(ax, [-0.2, FIELD_SIZE + 0.2]);
ylim(ax, [-0.2, FIELD_SIZE + 0.2]);
title(ax, 'Live Roboter Trajektorie');
xlabel(ax, 'X-Achse (Meter)');
ylabel(ax, 'Y-Achse (Meter)');

% --- Hintergrundbild auf das 3x3 m Feld mappen ---------------------------
% Das Bild wird ueber die Achsenkoordinaten gespannt: XData/YData geben an,
% auf welchen Meter-Bereich die Bildkanten gelegt werden ([0 3] x [0 3]).
%
% >>> Das Feldbild fe.jpg ist um 180 Grad gedreht -> wir korrigieren das
%     mit rot90(img, 2). Weitere Korrekturen fuer den Kollegen: <<<
%   Spiegeln waagerecht (links/rechts):  img = fliplr(img);
%   Spiegeln senkrecht  (oben/unten):    img = flipud(img);
%   Nur 90 Grad drehen:                  img = rot90(img);   % rot90(img,2)=180 Grad
%   Feld steht auf dem Kopf:             set(ax,'YDir','reverse');
try
    img = imread(BG_IMAGE);

    img = rot90(img, 2);   % 180-Grad-Korrektur des Hintergrundbildes
    img = fliplr(img);

    hImg = image(ax, 'CData', img, ...
                 'XData', [0, FIELD_SIZE], ...
                 'YData', [0, FIELD_SIZE]);
    uistack(hImg, 'bottom');     % Bild in den Hintergrund
    set(hImg, 'AlphaData', BG_ALPHA * ones(size(img,1), size(img,2)));

    % image() invertiert die Y-Achse standardmaessig -> auf mathematisch
    % (Y nach OBEN) zuruecksetzen. Bei Bedarf auf 'reverse' aendern.
    set(ax, 'YDir', 'normal');
catch ME
    warning(['Hintergrundbild "' BG_IMAGE '" konnte nicht geladen werden ' ...
             '(' ME.message '). Zeichne nur den Feldrahmen.']);
end

% Feldrahmen immer zeichnen (auch ohne Bild gut sichtbar)
rectangle(ax, 'Position', [0, 0, FIELD_SIZE, FIELD_SIZE], ...
          'EdgeColor', 'k', 'LineWidth', 3);

% -------------------------------------------------------------------------
% 3b. STATISCHE ELEMENTE: Parkbucht
% -------------------------------------------------------------------------
% Prediction-Boxen kommen jetzt LIVE ueber /global_obstacles (alpha < 1),
% daher hier keine statischen Platzhalter mehr.
drawParkingBay(ax, DRIVE_DIRECTION, FIELD_SIZE, PARK_CENTER_LON, ...
               PARK_GAP, PARK_WALL_LEN);

% -------------------------------------------------------------------------
% 3c. DYNAMISCHE GRAFIK-OBJEKTE
% -------------------------------------------------------------------------
% Trajektorie: normale Linie (kein animatedline), damit wir die geglaetteten
% Vektoren bei jedem Frame neu setzen koennen.
trajLine = plot(ax, NaN, NaN, 'b-', 'LineWidth', 2);
xHist = [];
yHist = [];

% Mitfahrendes Roboter-Koordinatensystem: X-Achse rot, Y-Achse gruen.
hAxisX = plot(ax, NaN, NaN, 'r-', 'LineWidth', 2.5);
hAxisY = plot(ax, NaN, NaN, 'g-', 'LineWidth', 2.5);

% Liste aller Hindernis-Grafiken (lokalisiert + Prediction). Wird bei jeder
% neuen MarkerArray komplett geloescht und neu gezeichnet -> verschwundene
% Predictions/Hindernisse bleiben nicht als Geister stehen.
obstaclePatches = gobjects(0);

% Flanken-Erkennung fuer das Reset-Signal (verhindert Ghost-Lines).
prevReset = false;

% -------------------------------------------------------------------------
% 4. LIVE-SCHLEIFE
% -------------------------------------------------------------------------
disp('Warte auf Live-Daten... (Schliesse das Grafikfenster zum Beenden)');
missedCount = 0;

while ishandle(fig)
    % --- 4a0. RESET-SIGNAL pruefen (Ghost-Line-Schutz) ------------------
    % obstacle_run.py sendet /global_reset == true waehrend des Einparkens.
    % Bei der steigenden Flanke (false->true) brechen wir die alte Trajektorie
    % ab, damit keine Linie quer ueber das Bild gezogen wird.
    try
        resetData = resetSub.LatestMessage;
    catch
        resetData = [];
    end
    if ~isempty(resetData)
        rFlag = getFieldCI(resetData, 'data');
        if ~isempty(rFlag) && logical(rFlag) && ~prevReset
            xHist = [];
            yHist = [];
            set(trajLine, 'XData', NaN, 'YData', NaN);   % Linie leeren
            disp('Reset empfangen -> Trajektorie zurueckgesetzt (Einparken).');
        end
        if ~isempty(rFlag)
            prevReset = logical(rFlag);
        end
    end

    % --- 4a. POSE empfangen ---------------------------------------------
    poseStatus = false;
    try
        [poseMsg, poseStatus] = receive(poseSub, RECEIVE_TIMEOUT);
    catch
        poseStatus = false;   % Timeout -> kein harter Fehler, weiter pollen
    end

    if poseStatus
        missedCount = 0;   % Verbindung lebt

        % Robuster Zugriff: Toolbox liefert je nach Version "pose"/"Pose".
        poseField = getFieldCI(poseMsg, 'pose');
        if ~isempty(poseField)
            posStruct = getFieldCI(poseField, 'position');
            x = getFieldCI(posStruct, 'x');
            y = getFieldCI(posStruct, 'y');

            if ~isempty(x) && ~isempty(y)
                x = double(x);  y = double(y);
                xHist(end+1) = x;   %#ok<AGROW>
                yHist(end+1) = y;   %#ok<AGROW>

                % --- Trajektorie glaetten (Moving Average) --------------
                % In den Kurven tracken wir nur die Frontwand -> die rohe
                % Linie ist eckig. smoothdata rundet die Bahn organisch.
                if numel(xHist) >= 2
                    xs = smoothdata(xHist, 'movmean', SMOOTH_WINDOW);
                    ys = smoothdata(yHist, 'movmean', SMOOTH_WINDOW);
                    set(trajLine, 'XData', xs, 'YData', ys);
                end

                % --- Mitfahrendes Koordinatensystem aus Yaw -------------
                oriStruct = getFieldCI(poseField, 'orientation');
                qz = getFieldCI(oriStruct, 'z');
                qw = getFieldCI(oriStruct, 'w');
                if ~isempty(qz) && ~isempty(qw)
                    yaw = 2.0 * atan2(double(qz), double(qw));
                    % 2D-Rotationsmatrix auf die Basisvektoren anwenden
                    R = [cos(yaw), -sin(yaw); sin(yaw), cos(yaw)];
                    exVec = R * [AXIS_LEN; 0];   % lokale X-Achse (rot)
                    eyVec = R * [0; AXIS_LEN];   % lokale Y-Achse (gruen)
                    set(hAxisX, 'XData', [x, x + exVec(1)], ...
                                'YData', [y, y + exVec(2)]);
                    set(hAxisY, 'XData', [x, x + eyVec(1)], ...
                                'YData', [y, y + eyVec(2)]);
                end
                drawnow limitrate;
            end
        end
    else
        missedCount = missedCount + 1;
        if missedCount == MAX_MISSED
            warning(['Seit ' num2str(MAX_MISSED * RECEIVE_TIMEOUT, '%.0f') ...
                     's keine Pose empfangen. Laeuft obstacle_run.py noch ' ...
                     'und stimmt die ROS_DOMAIN_ID (' ROS_DOMAIN ')?']);
        end
    end

    % --- 4b. HINDERNISSE + PREDICTIONS neu zeichnen ---------------------
    % Wir lesen nur den neuesten Stand und zeichnen ALLES neu (clear+redraw),
    % damit verschwundene Predictions/Hindernisse nicht stehen bleiben.
    % Unterscheidung lokalisiert/Prediction ueber color.a:
    %   a >= 1.0  -> lokalisiertes Hindernis (voll deckend)
    %   a <  1.0  -> Prediction-Box (halbtransparent, 25%)
    try
        latestObs = obsSub.LatestMessage;
    catch
        latestObs = [];
    end

    if ~isempty(latestObs)
        markers = getFieldCI(latestObs, 'markers');

        % Alte Hindernis-Grafiken entfernen
        delete(obstaclePatches(ishandle(obstaclePatches)));
        obstaclePatches = gobjects(0);

        if ~isempty(markers)
            for i = 1:numel(markers)
                m = markers(i);

                mPose  = getFieldCI(m, 'pose');
                mPos   = getFieldCI(mPose, 'position');
                x_obs  = getFieldCI(mPos, 'x');
                y_obs  = getFieldCI(mPos, 'y');
                mColor = getFieldCI(m, 'color');
                cr     = getFieldCI(mColor, 'r');
                ca     = getFieldCI(mColor, 'a');

                if isempty(x_obs) || isempty(y_obs)
                    continue;
                end

                % Farbe: rot wenn R-Kanal dominiert, sonst gruen
                if ~isempty(cr) && double(cr) >= 0.5
                    faceColor = [1 0 0];
                else
                    faceColor = [0 1 0];
                end

                % Deckkraft aus color.a (Prediction = transparent)
                if isempty(ca)
                    faceAlpha = 1.0;
                else
                    faceAlpha = double(ca);
                end

                halfSize = 0.075;  % Block ist ca. 15x15 cm
                cx = double(x_obs);  cy = double(y_obs);
                xs = [cx-halfSize, cx+halfSize, cx+halfSize, cx-halfSize];
                ys = [cy-halfSize, cy-halfSize, cy+halfSize, cy+halfSize];

                h = patch(ax, 'XData', xs, 'YData', ys, ...
                          'FaceColor', faceColor, 'FaceAlpha', faceAlpha, ...
                          'EdgeColor', 'k');
                obstaclePatches(end+1) = h;   %#ok<AGROW>
            end
            drawnow limitrate;
        end
    end
end

disp('Live-View beendet.');

% =========================================================================
% HILFSFUNKTIONEN
% =========================================================================
function val = getFieldCI(s, name)
% Robuster, case-insensitiver Feldzugriff. Die ROS-2-Toolbox benennt Felder
% je nach Version klein ("pose") oder gross ("Pose") -> wir suchen den
% Namen case-insensitiv. Liefert [] wenn das Feld (oder s) fehlt.
%
% Entspricht funktional dem gewuenschten Muster
%   if isfield(msg,'Pose'), pos = msg.Pose; else, pos = msg.pose; end
% deckt aber zusaetzlich beliebige Felder/Schreibweisen ab.
    val = [];
    if isempty(s) || ~isstruct(s)
        return;
    end
    fn = fieldnames(s);
    idx = find(strcmpi(fn, name), 1);
    if ~isempty(idx)
        val = s.(fn{idx});
    end
end

function drawParkingBay(ax, driveDir, fieldSize, centerLon, gap, wallLen)
% Zeichnet die zwei festen Parkwaende auf der Start-Geraden (Zone 0).
% Der Abstand zwischen den beiden Waenden ist exakt "gap" (22,25 cm).
% Die Waende ragen senkrecht von der Aussenbande ins Feld.
%
% >>> ANPASSEN: centerLon verschiebt die Bucht entlang der Geraden;
%     bei gespiegeltem Feld ggf. driveDir wechseln. <<<
%
% Geometrie je Fahrtrichtung (Segment 0):
%   'links'  -> Aussenbande im Osten (x = fieldSize), Fahrt entlang +Y
%   'rechts' -> Aussenbande im Westen (x = 0),        Fahrt entlang +Y
    yLow  = centerLon - gap/2;
    yHigh = centerLon + gap/2;

    if strcmpi(driveDir, 'links')
        xOuter = fieldSize;
        xInner = fieldSize - wallLen;
    else
        xOuter = 0;
        xInner = wallLen;
    end

    % Zwei Waende (Linien) mit exakt "gap" Abstand in Fahrtrichtung
    line(ax, [xOuter, xInner], [yLow,  yLow],  'Color', 'm', 'LineWidth', 3);
    line(ax, [xOuter, xInner], [yHigh, yHigh], 'Color', 'm', 'LineWidth', 3);
end
