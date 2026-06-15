% =========================================================================
% WRO - LIVE Visualisierung (ROS 2)
% =========================================================================
clear; clc; close all;

% 1. ROS-Umgebungsvariable setzen (anpassen falls nötig)
setenv('ROS_DOMAIN_ID', '0');

% 2. MATLAB als ROS 2 Node starten
try
    node = ros2node("/matlab_live_view");
    disp('ROS 2 Node erfolgreich gestartet.');
catch
    error('Konnte Node nicht starten. Läuft schon einer? (Nutze "clear" um alte Nodes zu löschen)');
end

% 1. Bild laden
img = imread('fe.jpg'); 

% 2. Bild passend für das 3x3 Meter Feld platzieren
% imshow(bild, [x_min, x_max, y_min, y_max])
% WICHTIG: Prüfe, ob dein Bild die richtige Ausrichtung hat (sonst ggf. rot90)
imshow(img, 'XData', [0 3], 'YData', [0 3]); 
hold on; 

% 3. Transparenz (Alpha-Wert) anpassen, damit man die blaue Linie gut sieht
set(gca, 'Children', flipud(get(gca, 'Children'))); % Bild in den Hintergrund
alpha(0.5); % Wert zwischen 0 (unsichtbar) und 1 (voll deckend)

% 3. Subscriber für unsere globalen Topics erstellen
disp('Verbinde mit /global_pose und /global_obstacles...');
poseSub = ros2subscriber(node, "/global_pose", "geometry_msgs/PoseStamped");
obsSub  = ros2subscriber(node, "/global_obstacles", "visualization_msgs/MarkerArray");

% 4. Das 3x3m Spielfeld zeichnen
figure('Name', 'WRO Live Map', 'Color', 'w');
hold on; grid on; axis equal;
xlim([-0.2 3.2]); ylim([-0.2 3.2]);
rectangle('Position', [0 0 3 3], 'EdgeColor', 'k', 'LineWidth', 3);
title('Live Roboter Trajektorie');
xlabel('X-Achse (Meter)'); ylabel('Y-Achse (Meter)');

% Animierte Linie für extrem schnelles Live-Zeichnen
trajLine = animatedline('Color', 'b', 'LineWidth', 2, 'Marker', '.');

% 5. Die Live-Schleife (Bricht ab, wenn man das Fenster schließt)
disp('Warte auf Live-Daten... (Schließe das Grafik-Fenster zum Beenden)');

while ishandle(gcf) % Läuft solange das Fenster offen ist
    
    % Versuche, eine Pose-Nachricht zu empfangen (Timeout = 0.5 Sekunden)
    [poseMsg, poseStatus] = receive(poseSub, 0.5);
    
    if poseStatus % Wenn eine neue Koordinate da ist
        x = poseMsg.pose.position.x;
        y = poseMsg.pose.position.y;
        
        % Punkt zur Linie hinzufügen
        addpoints(trajLine, x, y);
        drawnow limitrate; % Zeichnet das Bild neu, ohne MATLAB zu überlasten
    end
    
    % Optional: Hindernisse prüfen
    % Da Hindernisse seltener geupdatet werden, fragen wir einfach den
    % neuesten Stand ohne langes Blockieren ab:
    if obsSub.LatestMessage ~= false
        latestObs = obsSub.LatestMessage.Markers;
        for i = 1:length(latestObs)
            m = latestObs(i);
            x_obs = m.Pose.Position.X;
            y_obs = m.Pose.Position.Y;
            
            obsColor = 'g';
            if m.Color.R == 1.0; obsColor = 'r'; end
            
            % Zeichne/Überschreibe die Hindernisse
            rectangle('Position', [x_obs-0.075, y_obs-0.075, 0.15, 0.15], ...
                      'FaceColor', obsColor, 'EdgeColor', 'k');
        end
    end
end

disp('Live-View beendet.');