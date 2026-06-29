import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async redirects() {
    return [
      { source: '/bluetooth', destination: '/devices', permanent: true },
    ];
  },
};

export default nextConfig;
