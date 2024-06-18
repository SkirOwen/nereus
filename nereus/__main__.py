import nereus.config

from nereus import logger
from nereus import console


def main():
	console.rule("Nereus")

	args = nereus.config.parse_args()


if __name__ == "__main__":
	main()

