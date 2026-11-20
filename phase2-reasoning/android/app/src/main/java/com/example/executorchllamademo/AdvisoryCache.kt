
package com.example.executorchllamademo

object AdvisoryCache {

    private val cache = mutableMapOf<String, String>()

    fun get(key: String): String? {
        return cache[key]
    }

    fun put(key: String, response: String) {
        cache[key] = response
    }

    fun clear() {
        cache.clear()
    }
}
