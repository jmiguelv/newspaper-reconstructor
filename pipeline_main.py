"""CLI entry point for the jawi-pipeline article reconstruction module."""

from dotenv import load_dotenv

load_dotenv()

from src.newspaper_reconstructor.module import ArticleReconstructionModule

cli = ArticleReconstructionModule.make_cli()

if __name__ == "__main__":
    cli()
