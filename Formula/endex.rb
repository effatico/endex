# Homebrew formula for endex.
#
# Two ways to use it:
#   1. As a tap:  keep this file at Formula/endex.rb in a repo named
#      <user>/homebrew-endex, then:  brew install <user>/endex/endex
#   2. Directly:  brew install --build-from-source ./Formula/endex.rb
#
# After cutting a release, update `url` + `sha256` (and the tag in `version`).
# Generate the sha256 with:  shasum -a 256 endex-<target>.tar.gz

class Endex < Formula
  desc "Fast cached code indexer with MCP server for AI coding assistants"
  homepage "https://github.com/effatico/endex"
  version "0.1.2"
  license "MIT"

  on_macos do
    on_arm do
      url "https://github.com/effatico/endex/releases/download/v#{version}/endex-aarch64-apple-darwin.tar.gz"
      sha256 "0ceed0bcc810c86c18083af047053d133f5fcdab17fde95be30a7eb89ba10f14"
    end
    on_intel do
      url "https://github.com/effatico/endex/releases/download/v#{version}/endex-x86_64-apple-darwin.tar.gz"
      sha256 "bfcd8c6ca6847d8715ebd74b20d4bb68a97462aa485c65ecf81a65041cf2d87c"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/effatico/endex/releases/download/v#{version}/endex-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "2c7607ad43aa4fc32646b588e2b7fa5f0f485bac773e8b2ff6c81cbf6745b942"
    end
    on_intel do
      url "https://github.com/effatico/endex/releases/download/v#{version}/endex-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "10a70edd76c7038575dc0e768cc48e2eb366dd626057f4cdedb988c2e60f8f01"
    end
  end

  def install
    bin.install "endex"
  end

  def caveats
    <<~EOS
      Register endex as an MCP server for Claude Code:
        claude mcp add endex -- #{opt_bin}/endex mcp /path/to/your/repo

      For semantic search, add your provider env:
        claude mcp add endex \\
          -e EMBED_PROVIDER=openai \\
          -e EMBED_URL=http://localhost:11434/v1 \\
          -e EMBED_MODEL=qwen3-embedding \\
          -- #{opt_bin}/endex mcp /path/to/your/repo
    EOS
  end

  test do
    # The binary should respond to an MCP initialize handshake on stdio.
    output = pipe_output(bin/"endex", %Q[{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n mcp .\n], 5)
    assert_match "endex", output
  end
end
