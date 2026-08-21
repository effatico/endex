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
      sha256 "3377a58b29780b701fa8523c6ad90710b937246e06e9cd66db6b67b46ab5d84a"
    end
    on_intel do
      url "https://github.com/effatico/endex/releases/download/v#{version}/endex-x86_64-apple-darwin.tar.gz"
      sha256 "46ca2d81c89859246950244f1367e0ffc0c43377ccf70adef8ab4a75d5bfb557"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/effatico/endex/releases/download/v#{version}/endex-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "b478ef55e38e82f49ca781d0d82ad920829c11ed929e852b0fd3cb59ede48d12"
    end
    on_intel do
      url "https://github.com/effatico/endex/releases/download/v#{version}/endex-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "c7a41cd10189eac2637f7a5ebc7a71730c48bf9282b5d7605049cfd4fa6f08de"
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
