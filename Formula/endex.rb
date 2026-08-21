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
  version "0.2.0"
  license "MIT"

  on_macos do
    on_arm do
      url "https://github.com/effatico/endex/releases/download/v#{version}/endex-aarch64-apple-darwin.tar.gz"
      sha256 "4136b9c675c30a138b38c6e0b8c596e7b8d21483ac38016887941e1753743d86"
    end
    on_intel do
      url "https://github.com/effatico/endex/releases/download/v#{version}/endex-x86_64-apple-darwin.tar.gz"
      sha256 "ee36af335669c297b023517b5bfcd21e9917a8ade7d0abe9f803da3c19f35c6c"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/effatico/endex/releases/download/v#{version}/endex-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "73ea9224a5433fe222a3f88f981a64d52d9b629e6c23d43a4f48f191d601743d"
    end
    on_intel do
      url "https://github.com/effatico/endex/releases/download/v#{version}/endex-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "00f4b0aeafd3e631c9b57304cdfe63249c53392f4fc41b37da0eef240d8d5fd9"
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
    # The binary must answer an MCP initialize handshake on stdio when
    # invoked as `endex mcp DIR` (previously this only matched usage text).
    input = %({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n)
    output = pipe_output("#{bin}/endex mcp .", input, 10)
    assert_match '"serverInfo"', output
  end
end
