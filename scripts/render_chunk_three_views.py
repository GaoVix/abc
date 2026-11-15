"""Recorded real state RGB and real-calibrated action replays with two overlays."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import av
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, binary_erosion, map_coordinates
from scipy.spatial.transform import Rotation

from abc_minimal.export_mcap import ACTION_TOPICS, STATE_TOPICS, floor_indices
from scripts.verify_camera_fk_ik import Robot, pose, video_frames

ROOT = Path(__file__).resolve().parents[1]
JOINT_COLUMNS = [*range(6), *range(7, 13)]
ROWS = (
    ('real_state', 'old_state', 'new_state', '1  ORIGINAL REAL STATE | OLD IK state + NEW IK state'),
    ('real_action', 'old_action', 'old_on_old_prediction', '2  ORIGINAL REAL ACTION | OLD IK GT action + OLD prediction'),
    ('real_action', 'new_action', 'new_on_new_prediction', '3  ORIGINAL REAL ACTION | NEW IK GT action + NEW prediction'),
)
COLORS = ((40, 210, 255), (255, 165, 45))
ARM_GROUPS = {'left': 4, 'right': 5}


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def site_poses(robot):
    return np.stack([pose(robot.data.site(site)) for site in robot.sites])


def hide_camera_housings(robot):
    """Hide camera meshes/collision boxes while retaining all camera transforms."""
    hidden = []
    for geom in range(robot.model.ngeom):
        names = [robot.model.geom(geom).name]
        if robot.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            names.append(robot.model.mesh(int(robot.model.geom_dataid[geom])).name)
        body = int(robot.model.geom_bodyid[geom])
        while body:
            names.append(robot.model.body(body).name)
            body = int(robot.model.body_parentid[body])
        if any('camera' in name.lower() or 'd405' in name.lower() for name in names):
            # Group 3 already contains hidden collision geometry. Do not delete
            # camera bodies or touch their poses, FK, or the input trajectories.
            robot.model.geom_group[geom] = 3
            hidden.append(geom)
    robot.options.geomgroup[3] = 0
    return hidden


def prepare_arm_visibility(robot):
    """Separate visible arm geoms without enabling any hidden collision geoms."""
    visible = robot.options.geomgroup[robot.model.geom_group].astype(bool)
    if np.isin(robot.model.geom_group, list(ARM_GROUPS.values())).any():
        raise ValueError('Arm visibility groups are already in use')
    roots = {robot.model.body(side+'_arm').id: group for side,group in ARM_GROUPS.items()}
    for geom in np.flatnonzero(visible):
        body = int(robot.model.geom_bodyid[geom])
        while body and body not in roots:
            body = int(robot.model.body_parentid[body])
        if body in roots:
            robot.model.geom_group[geom] = roots[body]
    for group in ARM_GROUPS.values():
        robot.options.geomgroup[group] = 1


def select_view_arms(robot, camera):
    for side,group in ARM_GROUPS.items():
        robot.options.geomgroup[group] = camera not in ('left','right') or camera==side


def verify_view_visibility(scene, model, camera, camera_geoms):
    hidden_group = ARM_GROUPS['right' if camera=='left' else 'left'] if camera in ('left','right') else None
    for geom in scene.geoms[:scene.ngeom]:
        if geom.objtype==int(mujoco.mjtObj.mjOBJ_GEOM) and 0<=geom.objid<model.ngeom:
            assert geom.objid not in camera_geoms,'Camera housing leaked into the comparison render'
            if hidden_group is not None:
                assert model.geom_group[geom.objid]!=hidden_group,'Other arm leaked into a wrist view'


def pose_errors(actual, target):
    return [[float(1000*np.linalg.norm(a[:3, 3]-b[:3, 3])),
             float(np.rad2deg(Rotation.from_matrix(a[:3, :3]@b[:3, :3].T).magnitude()))]
            for a, b in zip(actual, target)]


def project_point(scene, point, width, height):
    """Project a world TCP through the same monocular GL camera as the RGB."""
    cameras = scene.camera
    position = np.mean([c.pos for c in cameras],axis=0)
    forward = np.mean([c.forward for c in cameras],axis=0)
    up = np.mean([c.up for c in cameras],axis=0)
    forward /= np.linalg.norm(forward)
    up /= np.linalg.norm(up)
    right = np.cross(forward,up)
    right /= np.linalg.norm(right)
    delta = np.asarray(point)-position
    depth = float(np.dot(delta,forward))
    near,far,bottom,top,center = [float(np.mean([getattr(c,name) for c in cameras]))
        for name in ('frustum_near','frustum_far','frustum_bottom','frustum_top','frustum_center')]
    if not near<depth<far:
        return None
    scale = 1. if cameras[0].orthographic else near/depth
    uv = np.array([width/2+height*(scale*np.dot(delta,right)-center)/(top-bottom)-.5,
                   height*(top-scale*np.dot(delta,up))/(top-bottom)-.5])
    if not np.isfinite(uv).all() or not (0<=uv[0]<width and 0<=uv[1]<height):
        return None
    return uv.tolist()


def draw_eef_alignment(rgb,scene,real_poses,overlay_poses,camera,recorded_intrinsics=None):
    """White cross = real; dataset-colored rings = the two XML TCPs."""
    image = Image.fromarray(rgb)
    pen = ImageDraw.Draw(image)
    height,width = rgb.shape[:2]
    visible_arms = [0] if camera=='left' else [1] if camera=='right' else [0,1]
    markers = {}
    project = (lambda p: project_recorded_point(scene,p,recorded_intrinsics)) if recorded_intrinsics else (
        lambda p: project_point(scene,p,width,height))
    for arm in visible_arms:
        real_uv = project(real_poses[arm,:3,3])
        circles = [project(p[arm,:3,3]) for p in overlay_poses]
        for uv,color,radius in zip(circles,COLORS,(9,13)):
            if uv is not None:
                x,y = uv
                box = (x-radius,y-radius,x+radius,y+radius)
                pen.ellipse(box,outline='#10131a',width=4)
                pen.ellipse(box,outline=color,width=2)
        if real_uv is not None:
            x,y = real_uv
            for color,stroke in (('#10131a',5),('white',3)):
                pen.line((x-7,y,x+7,y),fill=color,width=stroke)
                pen.line((x,y-7,x,y+7),fill=color,width=stroke)
        markers['left' if arm==0 else 'right'] = dict(real_cross=real_uv,
            cyan_circle=circles[0],orange_circle=circles[1])
    return image,markers


def shared_view(source, destination):
    """Both masks use the real solid's view, including its moving wrist cameras."""
    for original, overlay in zip(source.camera, destination.camera):
        for name in ('pos', 'forward', 'up'):
            getattr(overlay, name)[:] = getattr(original, name)
            np.testing.assert_array_equal(getattr(original, name), getattr(overlay, name))
        for name in ('frustum_center', 'frustum_bottom', 'frustum_top',
                     'frustum_near', 'frustum_far', 'orthographic'):
            setattr(overlay, name, getattr(original, name))
            assert getattr(overlay, name) == getattr(original, name)


def original_real_rows(args, metadata, ticks):
    """Read physical state/action; never substitute either processed IK dataset."""
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory
    calibrations = Path(getattr(args, 'calibrations', ROOT/'scripts/dishrack_joint_retarget_calibrations.json.gz'))
    record = next(r for r in json.load(gzip.open(calibrations, 'rt'))['records']
                  if r['episode'] == metadata['chunk']['source_episode'])
    raw_root = getattr(args, 'raw_root', None)
    if raw_root is None:
        raw_root = Path('/media/ps/T7/Datasets/ABC-130k/ABC-130k')
    path = Path(raw_root)/record['source_relative']
    if not path.is_file():
        raise FileNotFoundError(f'Original real MCAP required; set --raw-root: {path}')
    topics = STATE_TOPICS+ACTION_TOPICS
    values = {name: [] for name, _ in topics}
    cameras = (record['calibration']['camera_topic'], '/left-wrist-camera', '/right-wrist-camera')
    starts = {}
    with path.open('rb') as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, channel, message, value in reader.iter_decoded_messages(topics=[*values, *cameras], log_time_order=False):
            topic = channel.topic
            starts[topic] = min(starts.get(topic, message.log_time), message.log_time)
            if topic in values:
                values[topic].append((message.log_time, value.position))
    if set(starts) != set(values)|set(cameras):
        raise ValueError('Original episode lacks a required physical stream')
    chunk = metadata['chunk']
    expected = max(starts.values())+(np.arange(chunk['source_frame'], chunk['source_frame']+chunk['chunk_length'])+1)*33333333
    np.testing.assert_array_equal(expected, ticks)
    parts = []
    for topic, dim in topics:
        ordered = sorted(values[topic], key=lambda v: v[0])
        times = np.array([v[0] for v in ordered], dtype=np.int64)
        positions = np.asarray([v[1] for v in ordered], dtype=np.float64).reshape(-1, dim)
        parts.append(positions[floor_indices(times, ticks)])
    rows = np.concatenate(parts, axis=1)
    if not np.isfinite(rows).all():
        raise ValueError('Nonfinite original physical joints')
    return {'real_state': rows[:, :14], 'real_action': rows[:, 14:]}, record['calibration'], dict(
        mcap=str(path), sampling='unchanged official causal 30Hz; no interpolation/timeoffset/clipping',
        rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest(), calibrations_sha256=sha256(calibrations))


def read_recorded_views(path, calibration, ticks, cameras, nominal_fovys):
    """Decode unchanged RGB and camera info causally; keep native pixels intact."""
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory
    topics = {'top':calibration['camera_topic'],'left':'/left-wrist-camera','right':'/right-wrist-camera'}
    if any(camera not in topics for camera in cameras):
        raise ValueError('Recorded RGB requires top, left and right cameras')
    infos = {topics[c]+'-info':[] for c in cameras}
    states = {topic:[] for topic,_ in STATE_TOPICS}
    with Path(path).open('rb') as stream:
        reader = make_reader(stream,decoder_factories=[DecoderFactory()])
        for _,channel,message,value in reader.iter_decoded_messages(topics=[*infos,*states],log_time_order=False):
            if channel.topic in states:
                states[channel.topic].append((message.log_time,value.position))
            else:
                info = dict(width=value.width,height=value.height,distortion_model=value.distortion_model,
                            K=list(value.K),D=list(value.D))
                infos[channel.topic].append((message.log_time,info))
    states = {t:sorted(rows,key=lambda r:r[0]) for t,rows in states.items()}
    output = {}
    for camera in cameras:
        topic = topics[camera]
        camera_infos = sorted(infos[topic+'-info'],key=lambda r:r[0])
        info_indices = floor_indices(np.asarray([r[0] for r in camera_infos]),ticks) if camera_infos else None
        iterator = iter(video_frames(Path(path),topic))
        current = next(iterator)
        following = next(iterator,None)
        selected = []
        try:
            for index,tick in enumerate(ticks):
                while following is not None and following[0]<=tick:
                    current = following
                    following = next(iterator,None)
                if current[0]>tick:
                    raise ValueError(f'No causal RGB frame: {topic}')
                rgb = current[1].to_ndarray(format='rgb24')
                if camera_infos:
                    info_time,info = camera_infos[int(info_indices[index])]
                    assert info_time<=tick
                else:
                    if camera=='top':
                        info = copy.deepcopy(calibration['recorded_intrinsics'])
                        info['calibration_source'] = 'packed recorded top-camera intrinsics'
                    else:
                        height,width = rgb.shape[:2]
                        focal = height/(2*np.tan(np.deg2rad(nominal_fovys[camera])/2))
                        info = dict(width=width,height=height,distortion_model='none',D=[0.]*5,
                                    K=[focal,0.,(width-1)/2,0.,focal,(height-1)/2,0.,0.,1.],
                                    calibration_source='nominal XML pinhole; recorded camera info missing')
                assert rgb.shape==(info['height'],info['width'],3)
                # A wrist image's camera pose uses the measured joints at that
                # image timestamp. The plotted state/IK labels stay on the
                # unchanged training grid, with no interpolation/time offset.
                camera_state = []
                for state_topic,dim in STATE_TOPICS:
                    rows = states[state_topic]
                    j = int(floor_indices(np.asarray([r[0] for r in rows]),np.asarray([current[0]]))[0])
                    assert rows[j][0]<=current[0]
                    camera_state.extend(np.asarray(rows[j][1]).reshape(dim))
                selected.append(dict(rgb=rgb,timestamp_ns=int(current[0]),intrinsics=info,
                                     camera_state=np.asarray(camera_state,dtype=np.float64)))
        finally:
            iterator.close()
        output[camera] = selected
        print(f'Recorded RGB decoded: {camera}, {len(selected)} causal frames',flush=True)
    return output


def distort_normalized(x,y,intrinsics):
    """Match librealsense rs2_project_point_to_pixel for recorded camera models.

    https://github.com/realsenseai/librealsense/blob/master/src/rs.cpp
    """
    model = intrinsics['distortion_model'].lower()
    if model in ('','none') or not any(intrinsics['D']):
        return x,y
    if model not in ('inverse_brown_conrady','modified_brown_conrady','brown_conrady','plumb_bob'):
        raise ValueError(f'Unsupported recorded distortion: {model}')
    k1,k2,p1,p2,k3 = intrinsics['D']
    r2 = x*x+y*y
    radial = 1+((k3*r2+k2)*r2+k1)*r2
    xf,yf = x*radial,y*radial
    tx,ty = (xf,yf) if model in ('inverse_brown_conrady','modified_brown_conrady') else (x,y)
    return xf+2*p1*tx*ty+p2*(r2+2*tx*tx),yf+2*p2*tx*ty+p1*(r2+2*ty*ty)


def recorded_mask_map(intrinsics,fovy):
    """Map raw RGB pixels into ideal render pixels; never resample the RGB."""
    width,height = intrinsics['width'],intrinsics['height']
    k = np.asarray(intrinsics['K']).reshape(3,3)
    u,v = np.meshgrid(np.arange(width),np.arange(height))
    xd,yd = (u-k[0,2])/k[0,0],(v-k[1,2])/k[1,1]
    x,y = xd.copy(),yd.copy()
    # Invert the SDK's projection, including its inverse-Brown tangential
    # convention. Fixed-point convergence is checked over every image pixel.
    for _ in range(30):
        px,py = distort_normalized(x,y,intrinsics)
        x,y = x+(xd-px),y+(yd-py)
    px,py = distort_normalized(x,y,intrinsics)
    assert max(np.max(np.abs(px-xd)),np.max(np.abs(py-yd)))<1e-10
    focal = height/(2*np.tan(np.deg2rad(fovy)/2))
    return np.array([focal*y+(height-1)/2,focal*x+(width-1)/2])


def project_recorded_point(scene,point,intrinsics):
    position = np.mean([c.pos for c in scene.camera],axis=0)
    forward = np.mean([c.forward for c in scene.camera],axis=0)
    up = np.mean([c.up for c in scene.camera],axis=0)
    forward /= np.linalg.norm(forward)
    up /= np.linalg.norm(up)
    right = np.cross(forward,up)
    right /= np.linalg.norm(right)
    delta = np.asarray(point)-position
    depth = float(np.dot(delta,forward))
    if depth<=0:
        return None
    x,y = distort_normalized(np.dot(delta,right)/depth,-np.dot(delta,up)/depth,intrinsics)
    k = np.asarray(intrinsics['K']).reshape(3,3)
    uv = np.array([k[0,0]*x+k[0,2],k[1,1]*y+k[1,2]])
    if not np.isfinite(uv).all() or not (0<=uv[0]<intrinsics['width'] and 0<=uv[1]<intrinsics['height']):
        return None
    return uv.tolist()


def render_bundle(args, bundle, *, allow_fixture=False):
    bundle = Path(bundle)
    metadata = json.loads(bundle.with_suffix('.json').read_text())
    fixture = not isinstance(metadata.get('checkpoints'), dict)
    if fixture and not allow_fixture:
        raise ValueError('Actual old/new prediction bundle required; synthetic data is not model output')
    if not fixture and not metadata.get('exact_saved_labels'):
        raise ValueError('Bundle does not certify exact saved GT labels')
    chunk = metadata['chunk']
    count = int(chunk['chunk_length'])
    with np.load(bundle, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in {k for row in ROWS for k in row[1:3]}}
        ticks = saved['ticks_ns'].copy()
    if ticks.shape != (count,) or (np.diff(ticks) != 33333333).any():
        raise ValueError('Expected unchanged official 30Hz timestamps')
    real_arrays, calibration, source = original_real_rows(args, metadata, ticks)
    arrays.update(real_arrays)
    for key, value in arrays.items():
        if value.shape != (count, 14) or not np.isfinite(value).all():
            raise ValueError(f'Invalid trajectory: {key}')
    cameras = tuple(getattr(args, 'view_cameras', ('top', 'left', 'right')))
    if len(cameras)!=3 or len(set(cameras))!=3:
        raise ValueError('Select three distinct XML cameras')
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    if any((destination/name).exists() for name in ('comparison.mp4','comparison_rgb_lossless.mp4')):
        raise ValueError('Use a new render directory')
    model_path = ROOT/'abc_sim/models/yam_dishrack_base.xml'
    xml = Robot(model_path)
    mujoco.mj_forward(xml.model, xml.data)
    recorded = read_recorded_views(source['mcap'],calibration,ticks,cameras,
        {camera:float(xml.model.camera(camera).fovy[0]) for camera in cameras})
    world_from_camera = pose(xml.data.camera('top'))
    world_from_camera[:3, :3] @= np.diag([1.,-1.,-1.])
    camera_from_world = np.linalg.inv(world_from_camera)
    world_from_real_world = world_from_camera@np.asarray(calibration['left_base_to_camera'])
    reference_bases = {side: world_from_real_world@np.asarray(calibration['base_poses'][side])
                       for side in ('left','right')}
    solid = Robot(model_path, reference_bases)
    real_fk = Robot(ROOT/'abc_sim/models/yam_bimanual_empty.xml',calibration['base_poses'])
    original_model = mujoco.MjModel.from_xml_path(str(model_path))
    solid.model.geom_group[:] = original_model.geom_group
    solid.options = mujoco.MjvOption()
    solid.options.geomgroup[:] = 0
    solid.options.geomgroup[:3] = 1
    solid.options.sitegroup[:] = 0
    del original_model
    for robot in (solid,xml):
        robot.model.vis.quality.offsamples = 2
        robot.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        mujoco.mj_resetDataKeyframe(robot.model,robot.data,robot.model.key('home').id)
        mujoco.mj_forward(robot.model,robot.data)
        for camera in cameras:
            robot.model.camera(camera)
    for side in ('left','right'):
        np.testing.assert_allclose(camera_from_world@pose(solid.data.body(side+'_arm')),
            np.asarray(calibration['left_base_to_camera'])@np.asarray(calibration['base_poses'][side]),rtol=0,atol=2e-12)
    solid_camera_geoms = hide_camera_housings(solid)
    xml_camera_geoms = hide_camera_housings(xml)
    assert solid_camera_geoms == xml_camera_geoms
    # One union mask per data source: both arms share the same dataset color.
    arm_geoms = [g for g in range(xml.model.ngeom)
        if xml.model.body(int(xml.model.geom_bodyid[g])).name.startswith(('left_','right_'))
        and g not in xml_camera_geoms]
    for robot in (solid,xml):
        prepare_arm_visibility(robot)
    width,height = 640,480
    top,row_header,view_header,footer = 110,68,32,116
    panel_height = view_header+height+footer
    row_height = row_header+panel_height
    font = ImageFont.truetype('DejaVuSans.ttf',21)
    small = ImageFont.truetype('DejaVuSans.ttf',17)
    renderer = mujoco.Renderer(solid.model,height,width)
    segmenter = mujoco.Renderer(xml.model,height,width)
    segmenter.enable_segmentation_rendering()
    frames = []
    options = dict(codec='libx264rgb',pixelformat='rgb24',quality=None,macro_block_size=1,
                   output_params=['-crf','0','-preset','fast','-threads','1'])
    cached_movie = None
    cached_frames = None
    cached_report = None
    cached_path = getattr(args,'reuse_action_rows',None)
    if cached_path:
        cached_path = Path(cached_path)
        cached_report = json.loads((cached_path/'report.json').read_text())
        assert cached_report['bundle_sha256']==sha256(bundle) and cached_report['model_sha256']==sha256(model_path)
        assert cached_report['metadata']==metadata and cached_report['original_real_source']==source
        assert cached_report['frames']==count and cached_report['camera_housings']['hidden_in_both_overlays']
        cached_movie = av.open(str(cached_path/'comparison_rgb_lossless.mp4'))
        cached_frames = iter(cached_movie.decode(video=0))
    mask_maps = {}
    try:
        with imageio.get_writer(destination/'comparison_rgb_lossless.mp4',fps=30,**options) as video:
            for index in range(count):
                canvas = Image.new('RGB',(3*width,top+3*row_height),'#0c1119')
                if cached_frames is not None:
                    cached_rgb = next(cached_frames).to_ndarray(format='rgb24')
                    assert cached_rgb.shape==(canvas.height,canvas.width,3)
                    start = top+row_height
                    canvas.paste(Image.fromarray(cached_rgb[start:]),(0,start))
                draw = ImageDraw.Draw(canvas)
                prefix = 'SYNTHETIC TEST ONLY | ' if fixture else ''
                draw.text((12,8),f'{prefix}{chunk["source_episode"]} | frame {chunk["source_frame"]+index} | 30Hz',
                          font=font,fill='#ff7070' if fixture else 'white')
                draw.text((12,42),'Row 1: original real RGB | Rows 2/3: calibrated action replay | Two overlays: XML default bases | Color = dataset',
                          font=small,fill='#b4c9dc')
                draw.text((12,68),'EEF alignment: WHITE cross = original real TCP | CYAN circle = first overlay TCP | ORANGE circle = second overlay TCP',
                          font=small,fill='#b4c9dc')
                frame = dict(source_frame=chunk['source_frame']+index,rows=[])
                backgrounds = {}
                for row_number,(real_key,key_a,key_b,title) in enumerate(ROWS):
                    if row_number and cached_report is not None:
                        frame['rows'].append(copy.deepcopy(cached_report['metrics'][index]['rows'][row_number]))
                        continue
                    y = top+row_number*row_height
                    draw.text((12,y+7),title,font=font,fill='#f4d58a')
                    labels = ('OLD IK state','NEW IK state') if not row_number else (
                        'OLD IK action GT' if row_number==1 else 'NEW IK action GT',
                        'OLD model prediction' if row_number==1 else 'NEW model prediction')
                    draw.text((12,y+37),f'CYAN: {labels[0]}',font=small,fill=COLORS[0])
                    draw.text((600,y+37),f'ORANGE: {labels[1]}',font=small,fill=COLORS[1])
                    solid.set_state(arrays[real_key][index])
                    real_fk.set_state(arrays[real_key][index])
                    mujoco.mj_forward(solid.model,solid.data)
                    real_poses = site_poses(solid)
                    np.testing.assert_allclose(real_poses,world_from_real_world@site_poses(real_fk),rtol=0,atol=2e-12)
                    overlay_poses = []
                    for key in (key_a,key_b):
                        xml.set_state(arrays[key][index])
                        mujoco.mj_forward(xml.model,xml.data)
                        overlay_poses.append(site_poses(xml))
                    errors_a,errors_b = [pose_errors(p,real_poses) for p in overlay_poses]
                    paired_error = pose_errors(overlay_poses[1],overlay_poses[0])
                    joint_mae = float(np.rad2deg(np.abs(arrays[key_a][index,JOINT_COLUMNS]-arrays[key_b][index,JOINT_COLUMNS])).mean())
                    grip = float(np.abs(arrays[key_a][index,[6,13]]-arrays[key_b][index,[6,13]]).max())
                    views = []
                    for column,camera in enumerate(cameras):
                        x,panel_y = column*width,y+row_header
                        view_name = {'top':'TOP / BOTH ARMS','left':'LEFT WRIST / LEFT ARM ONLY',
                                     'right':'RIGHT WRIST / RIGHT ARM ONLY'}.get(camera,camera.upper())
                        if not row_number:
                            view_name = {'top':'TOP','left':'LEFT WRIST','right':'RIGHT WRIST'}[camera]+' / RECORDED REAL RGB'
                            if 'nominal' in recorded[camera][index]['intrinsics'].get('calibration_source',''):
                                view_name += ' (nominal K)'
                        draw.text((x+12,panel_y+5),view_name,font=small,fill='white')
                        recorded_view = recorded[camera][index] if not row_number else None
                        if recorded_view is not None:
                            solid.set_state(recorded_view['camera_state'])
                            mujoco.mj_forward(solid.model,solid.data)
                        select_view_arms(solid,camera)
                        renderer.update_scene(solid.data,camera=camera,scene_option=solid.options)
                        verify_view_visibility(renderer.scene,solid.model,camera,solid_camera_geoms)
                        rgb = recorded_view['rgb'].copy() if recorded_view is not None else renderer.render().copy()
                        assert rgb.shape==(height,width,3),'Recorded images must retain native resolution'
                        background_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
                        if (real_key,camera) in backgrounds:
                            assert background_hash==backgrounds[(real_key,camera)],'Action solid rows must be identical'
                        backgrounds[(real_key,camera)] = background_hash
                        masks = []
                        for key in (key_a,key_b):
                            xml.set_state(arrays[key][index])
                            mujoco.mj_forward(xml.model,xml.data)
                            select_view_arms(xml,camera)
                            segmenter.update_scene(xml.data,camera=camera,scene_option=xml.options)
                            verify_view_visibility(segmenter.scene,xml.model,camera,xml_camera_geoms)
                            shared_view(renderer.scene,segmenter.scene)
                            mask = np.isin(segmenter.render()[...,0],arm_geoms)
                            if recorded_view is not None:
                                info = recorded_view['intrinsics']
                                map_key = (camera,json.dumps(info,sort_keys=True))
                                if map_key not in mask_maps:
                                    mask_maps[map_key] = recorded_mask_map(info,float(solid.model.camera(camera).fovy[0]))
                                mask = map_coordinates(mask.astype(np.uint8),mask_maps[map_key],order=0,mode='constant',cval=0).astype(bool)
                            masks.append(mask)
                        for mask,color in zip(masks,COLORS):
                            rgb[mask] = np.rint(.88*rgb[mask]+.12*np.asarray(color)).astype(np.uint8)
                        # Outer/inner contour bands keep both colors visible for coincident poses.
                        rgb[binary_dilation(masks[0],iterations=2)&~masks[0]] = COLORS[0]
                        rgb[masks[1]&~binary_erosion(masks[1],iterations=2)] = COLORS[1]
                        marked,markers = draw_eef_alignment(rgb,renderer.scene,real_poses,overlay_poses,camera,
                            recorded_view['intrinsics'] if recorded_view is not None else None)
                        canvas.paste(marked,(x,panel_y+view_header))
                        if row_number==0:
                            lines = [(f'{side} {label}-real: {err[arm][0]:.4f} mm / {err[arm][1]:.4f} deg',color)
                                for err,label,color in ((errors_a,'OLD',COLORS[0]),(errors_b,'NEW',COLORS[1]))
                                for arm,side in enumerate(('L','R'))]
                        else:
                            lines = [(f'{side} pred-inputGT: {paired_error[arm][0]:.4f} mm / {paired_error[arm][1]:.4f} deg',COLORS[1])
                                     for arm,side in enumerate(('L','R'))]
                            lines += [(f'GT-real L {errors_a[0][0]:.3f}mm/{errors_a[0][1]:.3f}deg | R {errors_a[1][0]:.3f}mm/{errors_a[1][1]:.3f}deg',COLORS[0]),
                                      (f'Pred-GT joint MAE {joint_mae:.2f} deg | grip max {grip:.4f}','#c1cfdd')]
                        for line_number,(line,color) in enumerate(lines):
                            draw.text((x+12,panel_y+view_header+height+6+26*line_number),line,font=small,fill=color)
                        views.append(dict(camera=camera,solid_rgb_sha256=background_hash,
                            rgb_source='original_mcap_recorded_rgb' if recorded_view is not None else 'calibrated_action_sim_replay',
                            recorded_image_timestamp_ns=recorded_view['timestamp_ns'] if recorded_view is not None else None,
                            recorded_intrinsics=recorded_view['intrinsics'] if recorded_view is not None else None,
                            camera_pose_joint_timestamp='causal measured joints at recorded RGB timestamp' if recorded_view is not None else 'training action tick',
                            camera_extrinsics_source=('calibrated top-camera pose' if camera=='top' else 'model wrist camera mount at measured joints') if recorded_view is not None else 'XML camera on calibrated action replay',
                            visible_arms=[camera] if camera in ('left','right') else ['left','right'],
                            other_arm_hidden_in_solid_and_both_masks=recorded_view is None and camera in ('left','right'),
                            other_arm_hidden_in_both_masks=camera in ('left','right'),
                            real_gt_camera_position=solid.data.camera(camera).xpos.tolist(),
                            real_gt_camera_rotation=solid.data.camera(camera).xmat.reshape(3,3).tolist(),
                            both_masks_use_exact_solid_camera=True,
                            camera_housings_absent_from_solid_and_both_masks=True,
                            eef_alignment_pixels=markers,
                            masks=[dict(data=key,color_rgb=list(color),pixels=int(mask.sum()))
                                   for key,color,mask in zip((key_a,key_b),COLORS,masks)]))
                    frame['rows'].append(dict(solid=real_key,overlays=[key_a,key_b],
                        overlay_vs_real_error_mm_deg={key_a:errors_a,key_b:errors_b},
                        paired_xml_error_mm_deg=paired_error,paired_xml_joint_mae_deg=joint_mae,
                        paired_gripper_max_abs_error=grip,views=views))
                frame['action_rows_solid_rgb_identical'] = True
                video.append_data(np.asarray(canvas))
                if index==0:
                    canvas.save(destination/'comparison_preview.png')
                frames.append(frame)
                print(f'Two-overlay three-view render {index+1}/{count}',flush=True)
    finally:
        renderer.close()
        segmenter.close()
        if cached_movie is not None:
            cached_movie.close()
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(destination/'comparison_rgb_lossless.mp4'),
        '-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p,setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709',
        '-c:v','libx264','-preset','fast','-crf','12','-pix_fmt','yuv420p','-color_range','tv','-colorspace','bt709',
        '-color_primaries','bt709','-color_trc','bt709','-threads','2','-movflags','+faststart',str(destination/'comparison.mp4')],check=True)
    report = dict(status='complete',layout='three_rows_three_views_two_masks',test_fixture=fixture,
        actual_model_predictions=not fixture,bundle=str(bundle.resolve()),bundle_sha256=sha256(bundle),metadata=metadata,
        original_real_source=source,frames=count,fps=30,panels=9,panel_image_size=[width,height],camera_names=list(cameras),
        first_row_rgb='original MCAP RGB; causal timestamp selection; native pixels; no resize/undistortion/reconstruction',
        first_row_mask_projection='recorded per-camera K/D when available; missing wrist K explicitly marked nominal XML pinhole; raw-pixel remapping follows librealsense projection',
        first_row_camera_time='wrist camera extrinsics use measured joints at causal RGB timestamp; labels retain training ticks',
        reused_action_rows_from=str(cached_path) if cached_path else None,
        render_model=str(model_path),model_sha256=sha256(model_path),solid_base_mode='per_sequence_real_calibration_6d',
        overlay_base_mode='xml_default_6d',colors_identify='data source, shared by both arms',
        contour_bands='cyan outer 2px; orange inner 2px; both visible when silhouettes coincide',
        reference_base_poses_world={side:p.tolist() for side,p in reference_bases.items()},
        camera_from_real_base={side:(camera_from_world@p).tolist() for side,p in reference_bases.items()},
        camera_from_xml_base={side:(camera_from_world@pose(xml.data.body(side+'_arm'))).tolist() for side in ('left','right')},
        action_solid_identical_between_rows=True,all_masks_share_real_solid_camera=True,
        arm_visibility='Recorded RGB untouched; top overlays: both arms; wrist overlays: corresponding arm only; action replay matches overlay arm visibility',
        camera_housings=dict(hidden_in_solid=True,hidden_in_both_overlays=True,
            hidden_geom_ids=xml_camera_geoms,
            hidden_body_names=sorted({xml.model.body(int(xml.model.geom_bodyid[g])).name for g in xml_camera_geoms}),
            checked_every_rendered_scene=True,camera_poses_unchanged=True),
        eef_alignment=dict(real='white cross',first_overlay='cyan circle, radius 9px',
            second_overlay='orange circle, radius 13px',projection='actual RGB GL camera; invisible/out-of-frame TCPs omitted',
            arms='same visibility as the RGB and masks'),
        no_joint_clipping=True,no_new_ik=True,no_new_policy_inference=True,physics_steps=0,metrics=frames)
    (destination/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(destination/'comparison.mp4',flush=True)
