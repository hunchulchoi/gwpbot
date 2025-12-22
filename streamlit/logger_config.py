import logging
import sys

def setup_logging():
    """
    애플리케이션의 공통 로깅을 설정합니다.
    - 스트림 핸들러를 사용하여 콘솔에 로그를 출력합니다.
    - 모든 모듈에서 동일한 로거 설정을 공유할 수 있습니다.
    """
    # 이미 핸들러가 설정되어 있는지 확인하여 중복 설정을 방지합니다.
    if logging.getLogger().handlers:
        return

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout  # Streamlit 환경에서는 stdout으로 출력하는 것이 좋습니다.
    )