"""Request one low-speed gripper action from an already running A service."""
import pickle


def recover(side, action):
    import zmq
    if side not in ('left', 'right') or action not in ('open', 'close'):
        raise ValueError('夹爪操作无效')
    with zmq.Context() as context:
        with context.socket(zmq.REQ) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDTIMEO, 2000)
            socket.setsockopt(zmq.RCVTIMEO, 30000)
            socket.setsockopt(zmq.IMMEDIATE, 1)
            socket.connect('tcp://127.0.0.1:6001')
            try:
                socket.send(pickle.dumps(dict(method='recover_gripper', args=dict(side=side, action=action))))
                result = pickle.loads(socket.recv())
            except zmq.Again as exc:
                raise RuntimeError('终端 A 连接或响应超时，请检查 A 日志和实机夹爪状态，再决定下一次操作') from exc
    if isinstance(result, dict) and '__robot_rpc_error__' in result:
        if 'Invalid method' in result['__robot_rpc_error__']:
            raise RuntimeError('终端 A 尚未支持夹爪恢复，请正常退出并重新启动新版 A')
        raise RuntimeError(result['__robot_rpc_error__'])
    if not isinstance(result, dict) or 'position' not in result:
        raise RuntimeError('终端 A 尚未支持夹爪恢复，请正常退出并重新启动新版 A')
    return result['position']
