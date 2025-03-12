import os

from llmtuner.extras.misc import is_env_enabled
from llmtuner.webui.interface import create_ui


def main():
    gradio_ipv6 = is_env_enabled("GRADIO_IPV6")
    gradio_share = is_env_enabled("GRADIO_SHARE")
    server_name = os.getenv("GRADIO_SERVER_NAME", "[::]" if gradio_ipv6 else "0.0.0.0")
    create_ui().queue().launch(share=gradio_share, server_name=server_name, inbrowser=True)


if __name__ == "__main__":
    main()
