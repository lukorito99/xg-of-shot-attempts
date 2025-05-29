import tensorflow as tf
from collections import defaultdict, namedtuple
import cv2
import av
import numpy as np


#Uses MoviNet A3 [1,8,256,256,3]
# --- PyAV helper functions ---
VideoInfo = namedtuple("VideoInfo", ["total_frames", "fps"])

def get_video_info(video_path):
    """Retrieve video information (total frames and fps) using PyAV."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate is not None else 30.0
    if stream.frames and stream.frames > 0:
        total_frames = stream.frames
    else:
        # Fallback: calculate total frames using duration (container.duration is in microseconds)
        duration_sec = container.duration / 1e6 if container.duration is not None else 0
        total_frames = int(fps * duration_sec)
    container.close()
    return VideoInfo(total_frames=total_frames, fps=fps)

def get_video_frames_generator(video_path):
    """Generator that yields frames (in BGR format) from a video using PyAV."""
    container = av.open(video_path)
    for frame in container.decode(video=0):
        # Convert frame to BGR format to be consistent with cv2 usage
        yield frame.to_ndarray(format="bgr24")
    container.close()

def model_initialization(model_path):
    """
    Initialize MoviNet A3 stream for action recognition in football.

    Args:
        model_path (str): Path to the TFLite model file.

    Returns:
        tuple: (runner, states)
            - runner (tf.lite.Interpreter.SignatureRunner): TFLite model runner
            - states (dict): Initial states for the model
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    runner = interpreter.get_signature_runner()

    init_states = {
        name: tf.zeros(x['shape'], dtype=x['dtype'])
        for name, x in runner.get_input_details().items()
        if name != 'image'
    }
    states = init_states

    return runner, states

def preprocess_frame(frame):
    """
    Preprocess a frame for input into the TensorFlow Lite model.
    """
    frame = cv2.resize(frame, (256, 256))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    return tf.expand_dims(frame, axis=0)  # shape becomes [1,256,256,3]

def calculate_optical_flow(prev_frame, curr_frame):
    """
    Calculate optical flow between two frames.
    """
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(magnitude)

def inference_on_frames_stack(runner, frames, states):
    """
    Perform inference on a stack of 8 frames.

    Args:
        runner (tf.lite.Interpreter.SignatureRunner): TFLite model runner
        frames (list): List of 8 frames (each a numpy array).
        states (dict): Current states of the model.

    Returns:
        Tensor: Softmax probabilities from the model.
    """
    # Preprocess each frame so that each is [1,256,256,3]
    preprocessed_frames = [preprocess_frame(frame) for frame in frames]
    # Stack frames along a new temporal dimension so that input becomes [1,8,256,256,3]
    input_tensor = tf.concat(preprocessed_frames, axis=0)
    input_tensor = tf.expand_dims(input_tensor, axis=0)
    outputs = runner(**states, image=input_tensor)
    logits = outputs.pop('logits')[0]
    return tf.nn.softmax(logits)

def inference_on_video(
    video_path,
    tflite_model_path,
    window_size=8,
    stride=3,
    max_frame_skip=30,
    min_frame_skip=15,
    confidence_threshold=0.9,
    motion_threshold=0.5,
    progress_callback=None,
    segment_callback=None,
    abort_callback=None,
    pause_callback=None,
    state=None
):
    """
    Perform pausable inference on a video with overlapping windows.

    Returns a dict with:
      - actions: defaultdict(list) of detected 'ontarget' segments
      - state: the last internal state (for resuming)
      - completed: True if ran to end, False if paused or aborted
      - error: optional error string
    """
    # Always define actions so we can return it even on early errors
    actions = defaultdict(list)

    try:
        # --- Initialize or restore ---
        if state is None:
            state = {
                'initialized': False,
                'video_info': None,
                'frame_generator': None,
                'frame_count': 0,
                'frames_buffer': [],
                'current_action': None,
                'action_start_time': 0,
                'current_frame_skip': max_frame_skip,
                'prev_frame': None,
                'runner': None,
                'model_states': None,
                'classes': ['shortpass', 'longpass', 'throw', 'goalkick',
                            'penalty', 'corner', 'freekick', 'ontarget']
            }

            # First-time setup
            state['video_info'] = get_video_info(video_path)
            state['frame_generator'] = get_video_frames_generator(video_path)
            runner, init_states = model_initialization(tflite_model_path)
            state['runner'] = runner
            state['model_states'] = init_states
            state['initialized'] = True

        else:
            # Restore generator position safely
            if state.get('frame_count', 0) > 0:
                try:
                    for _ in range(state['frame_count']):
                        next(state['frame_generator'])
                except StopIteration:
                    # We've reached the end—treat as completed
                    return {
                        'actions': actions,
                        'state': state,
                        'completed': True
                    }

        # Unpack for readability
        video_info      = state['video_info']
        frame_generator = state['frame_generator']
        frame_count     = state['frame_count']
        frames_buffer   = state['frames_buffer']
        current_action  = state['current_action']
        action_start    = state['action_start_time']
        current_skip    = state['current_frame_skip']
        prev_frame      = state['prev_frame']
        runner          = state['runner']
        model_states    = state['model_states']
        classes         = state['classes']

        total_frames = video_info.total_frames
        fps          = video_info.fps

        # --- Main loop ---
        for frame in frame_generator:
            # 1) abort?
            if abort_callback and abort_callback():
                return {
                    'actions': actions,
                    'state': state,
                    'completed': False
                }

            # 2) pause?
            if pause_callback and pause_callback():
                return {
                    'actions': actions,
                    'state': state,
                    'completed': False
                }

            frame_count += 1

            # 3) frame skipping
            if frame_count % current_skip != 0:
                continue

            # 4) motion detection
            if prev_frame is not None:
                motion = calculate_optical_flow(prev_frame, frame)
                if motion > motion_threshold:
                    current_skip = min_frame_skip
            prev_frame = frame.copy()
            frames_buffer.append(frame)

            # 5) inference when we have enough
            if len(frames_buffer) >= window_size:
                window = frames_buffer[:window_size]
                probs  = inference_on_frames_stack(runner, window, model_states)
                idx    = np.argmax(probs)
                conf   = probs[idx].numpy()
                name   = classes[idx]
                time_s = frame_count / fps

                if current_action != name:
                    if current_action == "ontarget":
                        actions['ontarget'].append((action_start, time_s))
                        if segment_callback:
                            segment_callback((action_start, time_s))
                    current_action = name
                    action_start  = time_s
                    current_skip  = max(min_frame_skip, current_skip // 2)
                else:
                    current_skip = min(max_frame_skip, current_skip + 1)

                if conf < confidence_threshold:
                    current_skip = max(min_frame_skip, current_skip - 2)

                # slide buffer
                frames_buffer = frames_buffer[stride:]

            # 6) progress
            if progress_callback and total_frames:
                pct = int((frame_count / total_frames) * 100)
                progress_callback(pct, f"Processing... {pct}%")

            # 7) save state for potential pause
            state.update({
                'frame_count': frame_count,
                'frames_buffer': frames_buffer,
                'current_action': current_action,
                'action_start_time': action_start,
                'current_frame_skip': current_skip,
                'prev_frame': prev_frame,
                'model_states': model_states
            })

        # --- after loop we’ve reached the end ---
        if current_action == "ontarget":
            final_time = total_frames / fps
            actions['ontarget'].append((action_start, final_time))
            if segment_callback:
                segment_callback((action_start, final_time))

        return {
            'actions': actions,
            'state': state,
            'completed': True
        }

    except Exception as e:
        # Any unexpected error
        return {
            'actions': actions,
            'state': state,
            'completed': False,
            'error': str(e)
        }




