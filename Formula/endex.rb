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
  version "0.3.0"
  license "MIT"

  on_macos do
    on_arm do
      url "https://github.com/effatico/endex/releases/download/v#{version}/endex-aarch64-apple-darwin.tar.gz"
      sha256 "e90edbba2e3ba83985c13c2e23254bb67b41d90d625cfb5f0052f31a72a62f63"
    end
    on_intel do
      url "https://github.com/effatico/endex/releases/download/v#{version}/endex-x86_64-apple-darwin.tar.gz"
      sha256 "7c6f31a80d8dd125da3a37148f7f80ff42e97b6f6bb3e65501110c117db2e377"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/effatico/endex/releases/download/v#{version}/endex-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "18294aa42102e775e7e3c0882c944631e55cf434c2393f2899bc4982ecfd193c"
    end
    on_intel do
      url "https://github.com/effatico/endex/releases/download/v#{version}/endex-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "b2cb655a1bcbe3fa1275d560655659d4afeb360598e9c05b8ebc8ecc0e59c53c"
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
