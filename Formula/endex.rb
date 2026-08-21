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
  version "0.1.0"
  license "MIT"

  on_macos do
    on_arm do
      url "https://github.com/effatico/endex/releases/download/v#{version}/endex-aarch64-apple-darwin.tar.gz"
      sha256 "REPLACE_WITH_AARCH64_APPLE_DARWIN_SHA256"
    end
    on_intel do
      url "https://github.com/effatico/endex/releases/download/v#{version}/endex-x86_64-apple-darwin.tar.gz"
      sha256 "REPLACE_WITH_X86_64_APPLE_DARWIN_SHA256"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/effatico/endex/releases/download/v#{version}/endex-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "REPLACE_WITH_AARCH64_LINUX_SHA256"
    end
    on_intel do
      url "https://github.com/effatico/endex/releases/download/v#{version}/endex-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "REPLACE_WITH_X86_64_LINUX_SHA256"
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
