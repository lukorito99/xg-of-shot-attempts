import math
import os
import traceback

import joblib
import numpy as np
import pandas as pd


class XGModel:
    """
   for xG prediction:
      - Loads scaler, model, and label encoder once.
      - Computes 35 features from pitch coordinates.
      - Predicts a single xG float in [0,1].
      - Handles both goals automatically.
    """

    def __init__(self, scaler_path: str, model_path: str, label_encoder_path: str = None):
        # Load scaler
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        print("[DEBUG] Scaler loaded from:", scaler_path)

        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = joblib.load(model_path)
        print("[DEBUG] Model loaded from:", model_path)

        # Load label encoder if provided
        self.label_encoder = None
        if label_encoder_path and os.path.exists(label_encoder_path):
            self.label_encoder = joblib.load(label_encoder_path)
            print("[DEBUG] Label encoder loaded from:", label_encoder_path)

        # Define zone boundaries (from your code)
        self.zone_boundaries = {
            'horizontal': [0, 20, 40, 60, 80, 100, 120],  # 6 zones
            'vertical': [0, 26.67, 53.33, 80]  # 3 zones
        }

        # Special zones of interest
        self.zone_14_left = {'x': (20, 40), 'y': (26.67, 53.33)}  # Left Zone 14
        self.zone_14_right = {'x': (80, 100), 'y': (26.67, 53.33)}  # Right Zone 14

    def _extract_coords(self, coord) -> tuple:
        if isinstance(coord, dict):
            try:
                return float(coord["x"]), float(coord["y"])
            except KeyError as e:
                raise ValueError(f"Coordinate dictionary is missing key: {e}")
        elif isinstance(coord, (tuple, list)) and len(coord) == 2:
            return float(coord[0]), float(coord[1])
        elif isinstance(coord, np.ndarray):
            flat = coord.flatten()
            if flat.size == 2:
                return float(flat[0]), float(flat[1])
            else:
                raise ValueError("Coordinate numpy array must have exactly two elements.")
        else:
            raise ValueError("Coordinate must be a dict, a tuple/list with two elements, or a numpy array of size 2.")

    def determine_target_goal(self, start_pos, end_pos=None):
        """
        Determine which goal the shot is targeting based on ball movement.
        Returns goal_center, left_post, right_post for the target goal.
        """
        sx, sy = start_pos

        # Define both goals based on your coordinate system
        # Left goal (at x=0)
        left_goal_center = np.array([0.0, 40.0], dtype=np.float32)
        left_goal_left_post = np.array([0.0, 30.0], dtype=np.float32)  # Top post
        left_goal_right_post = np.array([0.0, 50.0], dtype=np.float32)  # Bottom post

        # Right goal (at x=120)
        right_goal_center = np.array([120.0, 40.0], dtype=np.float32)
        right_goal_left_post = np.array([120.0, 30.0], dtype=np.float32)  # Top post
        right_goal_right_post = np.array([120.0, 50.0], dtype=np.float32)  # Bottom post

        # Calculate distances to both goals
        dist_to_left = np.linalg.norm(left_goal_center - np.array([sx, sy]))
        dist_to_right = np.linalg.norm(right_goal_center - np.array([sx, sy]))

        # If we have end position, use trajectory to determine target
        if end_pos is not None:
            ex, ey = end_pos
            # Check if ball is moving towards left goal (decreasing x) or right goal (increasing x)
            x_direction = ex - sx

            if x_direction < 0:  # Moving towards left goal
                target_goal = "left"
                goal_center = left_goal_center
                left_post = left_goal_left_post
                right_post = left_goal_right_post
            else:  # Moving towards right goal
                target_goal = "right"
                goal_center = right_goal_center
                left_post = right_goal_left_post
                right_post = right_goal_right_post
        else:
            # No end position, use nearest goal
            if dist_to_left < dist_to_right:
                target_goal = "left"
                goal_center = left_goal_center
                left_post = left_goal_left_post
                right_post = left_goal_right_post
            else:
                target_goal = "right"
                goal_center = right_goal_center
                left_post = right_goal_left_post
                right_post = right_goal_right_post

        print(f"[DEBUG] Target goal: {target_goal} (distances: left={dist_to_left:.1f}m, right={dist_to_right:.1f}m)")
        return goal_center, left_post, right_post

    def assign_tactical_zone(self, x, y):
        """Assign tactical zone (1-18) based on x,y coordinates"""
        # Determine horizontal zone (1-6)
        h_zone = 1
        for i, boundary in enumerate(self.zone_boundaries['horizontal'][1:], 1):
            if x <= boundary:
                h_zone = i
                break

        # Determine vertical zone (1-3)
        v_zone = 1
        for i, boundary in enumerate(self.zone_boundaries['vertical'][1:], 1):
            if y <= boundary:
                v_zone = i
                break

        # Calculate final zone (1-18)
        zone = (v_zone - 1) * 6 + h_zone
        return zone

    def is_zone_14(self, x, y):
        """Check if position is in Zone 14 (either side)"""
        left_14 = (self.zone_14_left['x'][0] <= x <= self.zone_14_left['x'][1] and
                   self.zone_14_left['y'][0] <= y <= self.zone_14_left['y'][1])
        right_14 = (self.zone_14_right['x'][0] <= x <= self.zone_14_right['x'][1] and
                    self.zone_14_right['y'][0] <= y <= self.zone_14_right['y'][1])
        return left_14 or right_14

    def calculate_goal_mouth_accuracy(self, start_x, start_y, end_x, end_y):
        """Calculate how accurately the shot was aimed at goal center"""
        # Determine target goal based on shot direction
        if end_x < start_x:  # Shooting towards left goal
            target_center_y = 40
        else:  # Shooting towards right goal
            target_center_y = 40

        # Distance from goal center (vertically)
        accuracy = abs(end_y - target_center_y)
        return accuracy

    def validate_pitch_coordinates(self, x, y, point_name=""):
        """Validate that coordinates are within reasonable pitch bounds."""
        MIN_X, MAX_X = -2.0, 122.0  # Allow some margin beyond pitch
        MIN_Y, MAX_Y = -2.0, 82.0  # Allow some margin beyond pitch

        issues = []
        if not (MIN_X <= x <= MAX_X):
            issues.append(f"X coordinate {x:.2f} outside pitch bounds [{MIN_X}, {MAX_X}]")
        if not (MIN_Y <= y <= MAX_Y):
            issues.append(f"Y coordinate {y:.2f} outside pitch bounds [{MIN_Y}, {MAX_Y}]")

        if issues:
            print(f"[COORD WARNING] {point_name}: {', '.join(issues)}")
            return False
        return True

    def compute_all_features(self, start_pitch, end_pitch) -> pd.DataFrame:
        """
        Compute all 35 features used in training.
        """
        print("[DEBUG] Computing all features for start_pitch:", start_pitch,
              "end_pitch:", end_pitch)

        sx, sy = self._extract_coords(start_pitch)
        ex, ey = self._extract_coords(end_pitch)

        # Validate coordinates
        self.validate_pitch_coordinates(sx, sy, "Start position")
        self.validate_pitch_coordinates(ex, ey, "End position")

        start = np.array([sx, sy], dtype=np.float32)
        end = np.array([ex, ey], dtype=np.float32)

        # Determine target goal and get goal positions
        goal_center, left_post, right_post = self.determine_target_goal((sx, sy), (ex, ey))

        # Basic features
        distance_to_goal = np.linalg.norm(goal_center - start)
        ld = np.linalg.norm(left_post - start)
        rd = np.linalg.norm(right_post - start)
        gw = np.linalg.norm(left_post - right_post)

        if ld == 0 or rd == 0:
            angle_to_goal = 0.0
        else:
            cos_a = (ld ** 2 + rd ** 2 - gw ** 2) / (2 * ld * rd)
            cos_a = max(min(cos_a, 1.0), -1.0)
            angle_to_goal = math.degrees(math.acos(cos_a))

        shot_displacement = np.linalg.norm(end - start)
        shot_trajectory = math.degrees(math.atan2(ey - sy, ex - sx))

        # Tactical zone features
        tactical_zone = self.assign_tactical_zone(sx, sy)
        is_zone_14_val = int(self.is_zone_14(sx, sy))

        # Central corridor (middle third vertically)
        is_central_corridor = int(26.67 <= sy <= 53.33)

        # Penalty box features
        is_in_left_penalty_box = int(sx <= 18 and 18 <= sy <= 62)
        is_in_right_penalty_box = int(sx >= 102 and 18 <= sy <= 62)

        # Six-yard box features
        is_in_left_six_yard = int(sx <= 6 and 30 <= sy <= 50)
        is_in_right_six_yard = int(sx >= 114 and 30 <= sy <= 50)

        # Distance to penalty spots
        distance_to_left_penalty_spot = np.sqrt((sx - 12) ** 2 + (sy - 40) ** 2)
        distance_to_right_penalty_spot = np.sqrt((sx - 108) ** 2 + (sy - 40) ** 2)

        # Wing position indicators
        is_left_wing = int(sy <= 26.67)
        is_right_wing = int(sy >= 53.33)

        # Shot precision (distance to goal centers)
        shot_precision_left = np.sqrt((ex - 0) ** 2 + (ey - 40) ** 2)
        shot_precision_right = np.sqrt((ex - 120) ** 2 + (ey - 40) ** 2)

        # Goal mouth accuracy
        goal_mouth_accuracy = self.calculate_goal_mouth_accuracy(sx, sy, ex, ey)

        # Shot power proxy
        shot_power_proxy = shot_displacement

        # Quality metrics (normalized)
        max_angle = 180.0  # Assuming max angle is 180 degrees
        angle_quality = angle_to_goal / max_angle

        max_distance = 150.0  # Assuming max distance is ~150m (diagonal of pitch)
        distance_quality = 1 - (distance_to_goal / max_distance)

        # Combined position quality score
        position_quality = (angle_quality * 0.4 + distance_quality * 0.4 + is_zone_14_val * 0.2)

        # Interaction features
        distance_angle_interaction = distance_to_goal * angle_to_goal
        zone14_distance_interaction = is_zone_14_val * distance_to_goal
        penalty_box_angle = (is_in_left_penalty_box + is_in_right_penalty_box) * angle_to_goal

        # Shot direction categorization - FIXED: Use single categorical feature
        if -180 <= shot_trajectory <= -90:
            shot_direction_category = 'Sharp_Left'
        elif -90 < shot_trajectory <= -30:
            shot_direction_category = 'Left'
        elif -30 < shot_trajectory <= 30:
            shot_direction_category = 'Center'
        elif 30 < shot_trajectory <= 90:
            shot_direction_category = 'Right'
        elif 90 < shot_trajectory <= 180:
            shot_direction_category = 'Sharp_Right'
        else:
            shot_direction_category = 'Center'  # Default

        # One-hot encode direction features (as expected by the trained model)
        direction_Sharp_Left = int(shot_direction_category == 'Sharp_Left')
        direction_Left = int(shot_direction_category == 'Left')
        direction_Center = int(shot_direction_category == 'Center')
        direction_Right = int(shot_direction_category == 'Right')
        direction_Sharp_Right = int(shot_direction_category == 'Sharp_Right')

        # Create feature dictionary
        features = {
            # Basic coordinates and metrics
            'starting_x': sx,
            'starting_y': sy,
            'end_x': ex,
            'end_y': ey,
            'distance_to_goal': distance_to_goal,
            'angle_to_goal': angle_to_goal,
            'shot_displacement': shot_displacement,
            'shot_trajectory': shot_trajectory,

            # Tactical features
            'tactical_zone': tactical_zone,
            'is_zone_14': is_zone_14_val,
            'is_central_corridor': is_central_corridor,
            'is_in_left_penalty_box': is_in_left_penalty_box,
            'is_in_right_penalty_box': is_in_right_penalty_box,
            'is_in_left_six_yard': is_in_left_six_yard,
            'is_in_right_six_yard': is_in_right_six_yard,
            'distance_to_left_penalty_spot': distance_to_left_penalty_spot,
            'distance_to_right_penalty_spot': distance_to_right_penalty_spot,
            'is_left_wing': is_left_wing,
            'is_right_wing': is_right_wing,
            'shot_precision_left': shot_precision_left,
            'shot_precision_right': shot_precision_right,
            'goal_mouth_accuracy': goal_mouth_accuracy,
            'shot_power_proxy': shot_power_proxy,
            'angle_quality': angle_quality,
            'distance_quality': distance_quality,
            'position_quality': position_quality,

            # Interaction features
            'distance_angle_interaction': distance_angle_interaction,
            'zone14_distance_interaction': zone14_distance_interaction,
            'penalty_box_angle': penalty_box_angle,

            # Categorical direction feature (will be label encoded)
            'shot_direction_category': shot_direction_category,

            # One-hot encoded direction features (as expected by trained model)
            'direction_Sharp_Left': direction_Sharp_Left,
            'direction_Left': direction_Left,
            'direction_Center': direction_Center,
            'direction_Right': direction_Right,
            'direction_Sharp_Right': direction_Sharp_Right
        }

        # Create DataFrame with single row
        df = pd.DataFrame([features])

        print(f"[DEBUG] Computed {len(features)} features")
        print("[DEBUG] Feature values:", features)

        return df

    def predict(self, start_pitch, end_pitch) -> float:
        """
        Predict xG using all features with proper categorical encoding.
        """
        try:
            # Compute all features
            feature_df = self.compute_all_features(start_pitch, end_pitch)

            print("[DEBUG] Feature DataFrame shape:", feature_df.shape)
            print("[DEBUG] Feature DataFrame:\n", feature_df.head())

            if self.label_encoder is not None:
                categorical_columns = ['shot_direction_category']
                for col in categorical_columns:
                    if col in feature_df.columns and col in self.label_encoder:
                        try:
                            # Handle unseen categories by using the most common category
                            original_value = feature_df[col].iloc[0]
                            if original_value in self.label_encoder[col].classes_:
                                feature_df[col] = self.label_encoder[col].transform(feature_df[col])
                            else:
                                print(
                                    f"[WARNING] Unseen category '{original_value}' for {col}, using 'Center' as default")
                                feature_df[col] = self.label_encoder[col].transform(['Center'])
                        except Exception as e:
                            print(f"[ERROR] Failed to encode {col}: {e}")
                            # Use default encoding (Center = most common)
                            feature_df[col] = self.label_encoder[col].transform(['Center'])

            # Scale features
            scaled_features = self.scaler.transform(feature_df)
            print("[DEBUG] Scaled features shape:", scaled_features.shape)

            # Make prediction
            xg_val = float(self.model.predict(scaled_features)[0])
            print("[DEBUG] Raw xG prediction:", xg_val)

            # Additional debugging for suspicious values
            distance = feature_df['distance_to_goal'].iloc[0]
            if distance > 50:
                print(f"[WARNING] Very long shot distance: {distance:.2f}m")
            elif distance < 6:
                print(f"[INFO] Close range shot: {distance:.2f}m")

            if xg_val < 0.01:
                print(f"[INFO] Low xG ({xg_val:.6f}) for shot from {distance:.1f}m away")
            elif xg_val > 0.5:
                print(f"[INFO] High xG ({xg_val:.6f}) for shot from {distance:.1f}m away")

            return max(0.0, min(1.0, xg_val))

        except Exception as e:
            print(f"[DEBUG] Error in XGModel.predict: {e}")
            traceback.print_exc()
            return None

    def predict_batch(self, shots_data) -> list:
        """
        Predict xG for multiple shots.
        shots_data: list of tuples [(start_pos, end_pos), ...]
        """
        predictions = []
        for start_pos, end_pos in shots_data:
            pred = self.predict(start_pos, end_pos)
            predictions.append(pred)
        return predictions
